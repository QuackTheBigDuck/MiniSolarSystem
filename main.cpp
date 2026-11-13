#include <omp.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>

#ifdef _WIN32
#ifndef _WIN32_IE
#define _WIN32_IE 0x0500
#endif
#include <windows.h>
#include <commctrl.h>
#endif

struct Bodies {
    size_t n = 0;
    std::unique_ptr<double[]> x, y, z, vx, vy, vz, ax, ay, az, m;

    explicit Bodies(size_t n_)
        : n(n_),
          x(std::make_unique<double[]>(n_)),
          y(std::make_unique<double[]>(n_)),
          z(std::make_unique<double[]>(n_)),
          vx(std::make_unique<double[]>(n_)),
          vy(std::make_unique<double[]>(n_)),
          vz(std::make_unique<double[]>(n_)),
          ax(std::make_unique<double[]>(n_)),
          ay(std::make_unique<double[]>(n_)),
          az(std::make_unique<double[]>(n_)),
          m(std::make_unique<double[]>(n_)) {}
};

constexpr double G = 1.0;
constexpr double SOFTENING = 0.01;

void init_disk(Bodies& b, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    const double M_central = 100.0;
    std::uniform_real_distribution<double> planet_mass(0.005, 0.05);

    b.m[0] = M_central;
    b.x[0] = b.y[0] = b.z[0] = 0.0;
    b.vx[0] = b.vy[0] = b.vz[0] = 0.0;

    for (size_t i = 1; i < b.n; ++i) {
        double r = 0.8 + 7.2 * std::sqrt(U(rng));
        double th = 2.0 * M_PI * U(rng);
        b.m[i] = planet_mass(rng);
        double vc = std::sqrt(G * (M_central + b.m[i]) / r);

        b.x[i] = r * std::cos(th);
        b.y[i] = r * std::sin(th);
        b.z[i] = 0.05 * N01(rng);
        b.vx[i] = -vc * std::sin(th) + 0.02 * vc * N01(rng);
        b.vy[i] =  vc * std::cos(th) + 0.02 * vc * N01(rng);
        b.vz[i] = 0.02 * vc * N01(rng);
    }

    double px = 0, py = 0, pz = 0, mt = 0;
    for (size_t i = 0; i < b.n; ++i) {
        px += b.m[i] * b.vx[i]; py += b.m[i] * b.vy[i]; pz += b.m[i] * b.vz[i];
        mt += b.m[i];
    }
    for (size_t i = 0; i < b.n; ++i) {
        b.vx[i] -= px / mt; b.vy[i] -= py / mt; b.vz[i] -= pz / mt;
    }
}

void compute_accel(Bodies& b) {
    const size_t n = b.n;
    const double eps2 = SOFTENING * SOFTENING;
    const double* __restrict x = b.x.get();
    const double* __restrict y = b.y.get();
    const double* __restrict z = b.z.get();
    const double* __restrict m = b.m.get();
    double* __restrict ax = b.ax.get();
    double* __restrict ay = b.ay.get();
    double* __restrict az = b.az.get();

#pragma omp parallel for schedule(static)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        double axi = 0, ayi = 0, azi = 0;
        const double xi = x[i], yi = y[i], zi = z[i];

#pragma omp simd reduction(+ : axi, ayi, azi)
        for (size_t j = 0; j < n; ++j) {
            double dx = x[j] - xi;
            double dy = y[j] - yi;
            double dz = z[j] - zi;
            double r2 = dx * dx + dy * dy + dz * dz + eps2;
            double inv_r = 1.0 / std::sqrt(r2);
            double s = G * m[j] * inv_r * inv_r * inv_r;
            axi += s * dx;
            ayi += s * dy;
            azi += s * dz;
        }
        ax[i] = axi;
        ay[i] = ayi;
        az[i] = azi;
    }
}

double total_energy(const Bodies& b) {
    const size_t n = b.n;
    const double eps2 = SOFTENING * SOFTENING;
    const double* __restrict x = b.x.get();
    const double* __restrict y = b.y.get();
    const double* __restrict z = b.z.get();
    const double* __restrict vx = b.vx.get();
    const double* __restrict vy = b.vy.get();
    const double* __restrict vz = b.vz.get();
    const double* __restrict m = b.m.get();
    double kin = 0.0, pot = 0.0;

#pragma omp parallel for reduction(+ : kin, pot) schedule(static)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        kin += 0.5 * m[i] *
               (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
        double p = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            double dx = x[j] - x[i];
            double dy = y[j] - y[i];
            double dz = z[j] - z[i];
            p -= G * m[i] * m[j] / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
        pot += p;
    }
    return kin + pot;
}

void step(Bodies& b, double dt) {
    const long n = static_cast<long>(b.n);
    const double h = 0.5 * dt;
    double* __restrict x = b.x.get();
    double* __restrict y = b.y.get();
    double* __restrict z = b.z.get();
    double* __restrict vx = b.vx.get();
    double* __restrict vy = b.vy.get();
    double* __restrict vz = b.vz.get();
    double* __restrict ax = b.ax.get();
    double* __restrict ay = b.ay.get();
    double* __restrict az = b.az.get();
    const double* __restrict m = b.m.get();
    const double eps2 = SOFTENING * SOFTENING;

#pragma omp parallel
    {
#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i) {
            vx[i] += ax[i] * h;
            vy[i] += ay[i] * h;
            vz[i] += az[i] * h;
            x[i] += vx[i] * dt;
            y[i] += vy[i] * dt;
            z[i] += vz[i] * dt;
        }

#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i) {
            double axi = 0.0;
            double ayi = 0.0;
            double azi = 0.0;
            const double xi = x[i];
            const double yi = y[i];
            const double zi = z[i];

#pragma omp simd reduction(+ : axi, ayi, azi)
            for (size_t j = 0; j < static_cast<size_t>(n); ++j) {
                const double dx = x[j] - xi;
                const double dy = y[j] - yi;
                const double dz = z[j] - zi;
                const double r2 = dx * dx + dy * dy + dz * dz + eps2;
                const double inv_r = 1.0 / std::sqrt(r2);
                const double s = G * m[j] * inv_r * inv_r * inv_r;
                axi += s * dx;
                ayi += s * dy;
                azi += s * dz;
            }
            ax[i] = axi;
            ay[i] = ayi;
            az[i] = azi;
        }

#pragma omp for schedule(static)
        for (long i = 0; i < n; ++i) {
            vx[i] += ax[i] * h;
            vy[i] += ay[i] * h;
            vz[i] += az[i] * h;
        }
    }
}

void write_snapshot(const Bodies& b, int frame) {
    char name[64];
    std::snprintf(name, sizeof name, "snapshots/frame_%04d.csv", frame);
    FILE* f = std::fopen(name, "w");
    if (!f) { std::perror(name); return; }
    std::fprintf(f, "x,y,z\n");
    for (size_t i = 0; i < b.n; ++i)
        std::fprintf(f, "%.6f,%.6f,%.6f\n", b.x[i], b.y[i], b.z[i]);
    std::fclose(f);
}

#ifdef _WIN32
struct Visualization {
    Bodies& bodies;
    std::unique_ptr<double[]> x;
    std::unique_ptr<double[]> y;
    std::unique_ptr<double[]> previous_x;
    std::unique_ptr<double[]> previous_y;
    std::unique_ptr<double[]> mass;
    size_t count;
    std::atomic<size_t> visible_count;
    HWND slider = nullptr;
    HWND speed_slider = nullptr;
    HWND zoom_slider = nullptr;
    HWND pause_button = nullptr;
    HWND count_label = nullptr;
    HWND speed_label = nullptr;
    HWND zoom_label = nullptr;
    std::atomic<double> speed_multiplier = 1.0;
    std::atomic<double> zoom = 1.0;
    std::atomic<bool> paused = false;
    std::uint64_t step = 0;
    std::mutex mutex;
    std::atomic<bool> stop = false;
};

void publish_positions(Visualization& view) {
    std::lock_guard<std::mutex> lock(view.mutex);
    for (size_t i = 0; i < view.count; ++i) {
        view.x[i] = view.bodies.x[i];
        view.y[i] = view.bodies.y[i];
        view.mass[i] = view.bodies.m[i];
        view.previous_x[i] = view.x[i];
        view.previous_y[i] = view.y[i];
        view.mass[i] = view.bodies.m[i];
    }
}

LRESULT CALLBACK visualization_window_proc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    auto* view = reinterpret_cast<Visualization*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));

    if (message == WM_NCCREATE) {
        auto* create = reinterpret_cast<CREATESTRUCT*>(lParam);
        view = static_cast<Visualization*>(create->lpCreateParams);
        SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(view));
    }

    switch (message) {
        case WM_CREATE: {
            view->slider = CreateWindowExA(
                0, TRACKBAR_CLASSA, "", WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS,
                0, 0, 0, 0, hwnd, reinterpret_cast<HMENU>(100),
                GetModuleHandle(nullptr), nullptr);
            SendMessage(view->slider, TBM_SETRANGEMIN, TRUE, 1);
            SendMessage(view->slider, TBM_SETRANGEMAX, TRUE,
                        static_cast<LPARAM>(view->count));
            SendMessage(view->slider, TBM_SETPOS, TRUE, static_cast<LPARAM>(view->count));
            view->speed_slider = CreateWindowExA(
                0, TRACKBAR_CLASSA, "", WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS,
                0, 0, 0, 0, hwnd, reinterpret_cast<HMENU>(103),
                GetModuleHandle(nullptr), nullptr);
            SendMessage(view->speed_slider, TBM_SETRANGE, TRUE, MAKELONG(1, 200));
            SendMessage(view->speed_slider, TBM_SETPOS, TRUE, 100);
            view->zoom_slider = CreateWindowExA(
                0, TRACKBAR_CLASSA, "", WS_CHILD | WS_VISIBLE | TBS_AUTOTICKS,
                0, 0, 0, 0, hwnd, reinterpret_cast<HMENU>(104),
                GetModuleHandle(nullptr), nullptr);
            SendMessage(view->zoom_slider, TBM_SETRANGE, TRUE, MAKELONG(25, 200));
            SendMessage(view->zoom_slider, TBM_SETPOS, TRUE, 100);
            view->count_label = CreateWindowExA(
                0, "STATIC", "", WS_CHILD | WS_VISIBLE | SS_RIGHT,
                0, 0, 0, 0, hwnd, reinterpret_cast<HMENU>(101),
                GetModuleHandle(nullptr), nullptr);
            view->speed_label = CreateWindowExA(
                0, "STATIC", "", WS_CHILD | WS_VISIBLE | SS_RIGHT,
                0, 0, 0, 0, hwnd, reinterpret_cast<HMENU>(105),
                GetModuleHandle(nullptr), nullptr);
            view->zoom_label = CreateWindowExA(
                0, "STATIC", "", WS_CHILD | WS_VISIBLE | SS_RIGHT,
                0, 0, 0, 0, hwnd, reinterpret_cast<HMENU>(106),
                GetModuleHandle(nullptr), nullptr);
            view->pause_button = CreateWindowExA(
                0, "BUTTON", "Pause", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0, hwnd, reinterpret_cast<HMENU>(102),
                GetModuleHandle(nullptr), nullptr);
            SendMessage(view->count_label, WM_SETFONT,
                        reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
            SendMessage(view->speed_label, WM_SETFONT,
                        reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
            SendMessage(view->zoom_label, WM_SETFONT,
                        reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
            SendMessage(view->pause_button, WM_SETFONT,
                        reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
            return 0;
        }

        case WM_SIZE: {
            const int width = LOWORD(lParam);
            const int height = HIWORD(lParam);
            const int panel_height = 112;
            const int padding = 18;
            if (view != nullptr && view->slider != nullptr && width > 500) {
                const int slider_width = (width - 180) / 3;
                const int row_y = height - 94;
                MoveWindow(view->slider, padding, row_y, slider_width, 28, TRUE);
                MoveWindow(view->speed_slider, padding + slider_width + 20, row_y,
                           slider_width, 28, TRUE);
                MoveWindow(view->zoom_slider, padding + (slider_width + 20) * 2, row_y,
                           slider_width, 28, TRUE);
                MoveWindow(view->count_label, padding, height - 62, slider_width, 24, TRUE);
                MoveWindow(view->speed_label, padding + slider_width + 20,
                           height - 62, slider_width, 24, TRUE);
                MoveWindow(view->zoom_label, padding + (slider_width + 20) * 2,
                           height - 62, slider_width, 24, TRUE);
                MoveWindow(view->pause_button, width - 78, height - 52, 62, 30, TRUE);
                InvalidateRect(hwnd, nullptr, FALSE);
            }
            return 0;
        }

        case WM_HSCROLL:
            if (view != nullptr && reinterpret_cast<HWND>(lParam) == view->slider) {
                const size_t selected = static_cast<size_t>(
                    SendMessage(view->slider, TBM_GETPOS, 0, 0));
                view->visible_count = selected;
                char text[64];
                std::snprintf(text, sizeof text, "Particles: %zu", selected);
                SetWindowTextA(view->count_label, text);
                InvalidateRect(hwnd, nullptr, FALSE);
            }
            if (view != nullptr && reinterpret_cast<HWND>(lParam) == view->speed_slider) {
                const int selected = static_cast<int>(
                    SendMessage(view->speed_slider, TBM_GETPOS, 0, 0));
                view->speed_multiplier = selected / 100.0;
                char text[64];
                std::snprintf(text, sizeof text, "Speed: %.1fx", selected / 100.0);
                SetWindowTextA(view->speed_label, text);
            }
            if (view != nullptr && reinterpret_cast<HWND>(lParam) == view->zoom_slider) {
                const int selected = static_cast<int>(
                    SendMessage(view->zoom_slider, TBM_GETPOS, 0, 0));
                view->zoom = selected / 100.0;
                char text[64];
                std::snprintf(text, sizeof text, "Zoom: %.1fx", selected / 100.0);
                SetWindowTextA(view->zoom_label, text);
                InvalidateRect(hwnd, nullptr, FALSE);
            }
            return 0;

        case WM_COMMAND:
            if (view != nullptr && LOWORD(wParam) == 102 && HIWORD(wParam) == BN_CLICKED) {
                const bool paused = !view->paused.load();
                view->paused = paused;
                SetWindowTextA(view->pause_button, paused ? "Resume" : "Pause");
            }
            return 0;

        case WM_TIMER:
            InvalidateRect(hwnd, nullptr, FALSE);
            return 0;

        case WM_ERASEBKGND:
            return 1;

        case WM_PAINT: {
            PAINTSTRUCT paint;
            HDC dc = BeginPaint(hwnd, &paint);
            RECT client;
            GetClientRect(hwnd, &client);
            const int panel_height = 112;
            const int canvas_height = std::max(1, static_cast<int>(client.bottom) - panel_height);
            const int width = std::max(1, static_cast<int>(client.right));
            const int height = std::max(1, static_cast<int>(client.bottom));

            HDC buffer = CreateCompatibleDC(dc);
            HBITMAP bitmap = CreateCompatibleBitmap(dc, width, height);
            HGDIOBJ old_bitmap = SelectObject(buffer, bitmap);
            HDC draw_dc = buffer;

            HBRUSH background = CreateSolidBrush(RGB(3, 5, 14));
            FillRect(draw_dc, &client, background);
            DeleteObject(background);

            if (view != nullptr) {
                std::lock_guard<std::mutex> lock(view->mutex);
                const double center_x = (client.right - client.left) * 0.5;
                const double center_y = canvas_height * 0.5;
                const double scale = std::min(static_cast<int>(client.right - client.left),
                                              canvas_height) / 18.0 * view->zoom.load();

                HPEN orbit_pen = CreatePen(PS_SOLID, 1, RGB(18, 24, 48));
                HGDIOBJ old_pen = SelectObject(draw_dc, orbit_pen);
                HGDIOBJ old_brush = SelectObject(draw_dc, GetStockObject(NULL_BRUSH));
                for (int ring = 2; ring <= 8; ++ring) {
                    const int radius = static_cast<int>(ring * scale);
                    Ellipse(draw_dc, static_cast<int>(center_x) - radius,
                            static_cast<int>(center_y) - radius,
                            static_cast<int>(center_x) + radius,
                            static_cast<int>(center_y) + radius);
                }
                SelectObject(draw_dc, old_brush);
                SelectObject(draw_dc, old_pen);
                DeleteObject(orbit_pen);

                HBRUSH white_star_brush = CreateSolidBrush(RGB(240, 245, 255));
                HBRUSH blue_star_brush = CreateSolidBrush(RGB(180, 210, 255));
                HBRUSH orange_star_brush = CreateSolidBrush(RGB(255, 190, 100));
                HPEN trail_pen = CreatePen(PS_SOLID, 1, RGB(35, 55, 100));
                old_brush = SelectObject(draw_dc, white_star_brush);

                const size_t visible_count = std::min(view->visible_count.load(), view->count);
                for (size_t i = 1; i < visible_count; ++i) {
                    const int px = static_cast<int>(center_x + view->x[i] * scale);
                    const int py = static_cast<int>(center_y - view->y[i] * scale);
                    const int previous_px = static_cast<int>(center_x + view->previous_x[i] * scale);
                    const int previous_py = static_cast<int>(center_y - view->previous_y[i] * scale);
                    HGDIOBJ old_particle_pen = SelectObject(draw_dc, trail_pen);
                    MoveToEx(draw_dc, previous_px, previous_py, nullptr);
                    LineTo(draw_dc, px, py);
                    SelectObject(draw_dc, old_particle_pen);

                    const double mass = view->mass[i];
                    HBRUSH star_brush = (i % 5 == 0) ? orange_star_brush
                        : (i % 3 == 0) ? blue_star_brush : white_star_brush;
                    HGDIOBJ old_star_brush = SelectObject(draw_dc, star_brush);
                    const int radius = std::clamp(
                        2 + static_cast<int>(std::sqrt(mass) * 10.0), 2, 5);
                    Ellipse(draw_dc, px - radius, py - radius, px + radius + 1, py + radius + 1);
                    SelectObject(draw_dc, old_star_brush);
                }

                SelectObject(draw_dc, old_brush);
                DeleteObject(white_star_brush);
                DeleteObject(blue_star_brush);
                DeleteObject(orange_star_brush);
                DeleteObject(trail_pen);
                HBRUSH sun_glow = CreateSolidBrush(RGB(120, 55, 15));
                HBRUSH sun = CreateSolidBrush(RGB(255, 190, 45));
                const int glow_radius = 18;
                const int sun_radius = 9;
                HGDIOBJ old_sun_brush = SelectObject(draw_dc, sun_glow);
                Ellipse(draw_dc, static_cast<int>(center_x) - glow_radius,
                        static_cast<int>(center_y) - glow_radius,
                        static_cast<int>(center_x) + glow_radius,
                        static_cast<int>(center_y) + glow_radius);
                SelectObject(draw_dc, sun);
                Ellipse(draw_dc, static_cast<int>(center_x) - sun_radius,
                        static_cast<int>(center_y) - sun_radius,
                        static_cast<int>(center_x) + sun_radius,
                        static_cast<int>(center_y) + sun_radius);
                SelectObject(draw_dc, old_sun_brush);
                DeleteObject(sun_glow);
                DeleteObject(sun);

            }

            BitBlt(dc, 0, 0, width, height, buffer, 0, 0, SRCCOPY);
            SelectObject(buffer, old_bitmap);
            DeleteObject(bitmap);
            DeleteDC(buffer);
            EndPaint(hwnd, &paint);
            return 0;
        }

        case WM_CLOSE:
            DestroyWindow(hwnd);
            return 0;

        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) DestroyWindow(hwnd);
            return 0;

        case WM_DESTROY:
            if (view != nullptr) view->stop = true;
            KillTimer(hwnd, 1);
            PostQuitMessage(0);
            return 0;
    }

    return DefWindowProc(hwnd, message, wParam, lParam);
}

int run_visualization(Bodies& bodies, double dt, int threads) {
    Visualization view{bodies, std::make_unique<double[]>(bodies.n),
                        std::make_unique<double[]>(bodies.n),
                        std::make_unique<double[]>(bodies.n),
                        std::make_unique<double[]>(bodies.n),
                        std::make_unique<double[]>(bodies.n),
                        bodies.n, bodies.n};
    publish_positions(view);

    std::thread simulation([&view, dt, threads] {
        omp_set_num_threads(threads);
        while (!view.stop) {
            const auto frame_start = std::chrono::steady_clock::now();
            while (view.paused && !view.stop) std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (view.stop) break;
            step(view.bodies, dt * view.speed_multiplier.load());
            {
                std::lock_guard<std::mutex> lock(view.mutex);
                for (size_t i = 0; i < view.count; ++i) {
                    view.previous_x[i] = view.x[i];
                    view.previous_y[i] = view.y[i];
                    view.x[i] = view.bodies.x[i];
                    view.y[i] = view.bodies.y[i];
                }
                ++view.step;
            }
            const auto frame_time = std::chrono::steady_clock::now() - frame_start;
            const auto frame_period = std::chrono::milliseconds(16);
            if (frame_time < frame_period) {
                std::this_thread::sleep_for(frame_period - frame_time);
            }
        }
    });

    const char* class_name = "NBodyVisualizationWindow";
    INITCOMMONCONTROLSEX common_controls{sizeof(INITCOMMONCONTROLSEX), ICC_BAR_CLASSES};
    InitCommonControlsEx(&common_controls);
    WNDCLASSA window_class{};
    window_class.lpfnWndProc = visualization_window_proc;
    window_class.style = CS_HREDRAW | CS_VREDRAW;
    window_class.hInstance = GetModuleHandle(nullptr);
    window_class.hCursor = LoadCursor(nullptr, IDC_ARROW);
    window_class.lpszClassName = class_name;
    RegisterClassA(&window_class);

    HWND window = CreateWindowExA(
        0, class_name, "N-body particles", WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
        CW_USEDEFAULT, CW_USEDEFAULT, 900, 900, nullptr, nullptr,
        window_class.hInstance, &view);
    if (window == nullptr) {
        view.stop = true;
        simulation.join();
        std::fprintf(stderr, "Could not create visualization window\n");
        return EXIT_FAILURE;
    }

    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);
    char initial_label[64];
    std::snprintf(initial_label, sizeof initial_label, "Particles: %zu", bodies.n);
    SetWindowTextA(view.count_label, initial_label);
    SetWindowTextA(view.speed_label, "Speed: 1.0x");
    SetWindowTextA(view.zoom_label, "Zoom: 1.0x");
    SetTimer(window, 1, 16, nullptr);

    MSG message;
    while (GetMessage(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessage(&message);
    }

    view.stop = true;
    simulation.join();
    return 0;
}
#endif

int main(int argc, char** argv) {
    size_t N        = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 10000;
    int steps       = argc > 2 ? std::atoi(argv[2]) : 200;
    double dt       = argc > 3 ? std::atof(argv[3]) : 0.001;
    int threads     = argc > 4 ? std::atoi(argv[4]) : omp_get_max_threads();
    int snap_every  = argc > 5 ? std::atoi(argv[5]) : 0;
#ifdef _WIN32
    const bool visualize = argc <= 6 || std::string(argv[6]) != "--headless";
#else
    const bool visualize = false;
#endif

    if (N == 0) {
        std::fprintf(stderr, "N must be greater than zero\n");
        return EXIT_FAILURE;
    }

    omp_set_num_threads(threads);
    std::printf("N=%zu steps=%d dt=%g threads=%d\n", N, steps, dt, threads);

    Bodies b(N);
    init_disk(b);
    compute_accel(b);

    const double E0 = total_energy(b);
    std::printf("Initial energy: %.8f\n", E0);

#ifdef _WIN32
    if (visualize) return run_visualization(b, dt, threads);
#endif

    auto t0 = std::chrono::steady_clock::now();
    for (int s = 1; s <= steps; ++s) {
        step(b, dt);
        if (snap_every > 0 && s % snap_every == 0) write_snapshot(b, s / snap_every);
        if (s % std::max(1, steps / 10) == 0) {
            double E = total_energy(b);
            std::printf("step %6d  E=%.8f  rel.drift=%.3e\n", s, E, std::abs((E - E0) / E0));
        }
    }
    auto t1 = std::chrono::steady_clock::now();

    double secs = std::chrono::duration<double>(t1 - t0).count();
    double interactions = static_cast<double>(N) * N * steps;
    std::printf("Time: %.3f s  (%.2f G interactions/s)\n", secs, interactions / secs / 1e9);
    return 0;
}