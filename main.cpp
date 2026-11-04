// Parallel N-body gravity simulation (O(N^2) direct summation) using OpenMP.
//
// Build:  g++ -O3 -march=native -fopenmp -std=c++17 nbody.cpp -o nbody
// Run:    untitled.exe [N=10000] [steps=200] [dt=0.001] [threads=all] [snapshot_every=0]
//         Add --headless as the sixth argument to run without the window.
//
// If snapshot_every > 0, positions are written to snapshots/frame_XXXX.csv
// (create the folder first: mkdir snapshots).
//
// Units: G = 1. A heavy central body with light particles on roughly
// circular orbits in a disk (a toy "solar system / galaxy").
// Integrator: kick-drift-kick leapfrog (symplectic, good energy behaviour).

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
    // Structure-of-arrays layout: each array is allocated contiguously.
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
constexpr double SOFTENING = 0.01;  // avoids singularities in close encounters

void init_disk(Bodies& b, unsigned seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N01(0.0, 1.0);

    const double M_central = 1000.0;
    const double m_small = 1.0 / b.n;

    // Body 0: central mass at rest
    b.m[0] = M_central;
    b.x[0] = b.y[0] = b.z[0] = 0.0;
    b.vx[0] = b.vy[0] = b.vz[0] = 0.0;

    for (size_t i = 1; i < b.n; ++i) {
        double r = 0.5 + 4.5 * std::sqrt(U(rng));  // radius in [0.5, 5]
        double th = 2.0 * M_PI * U(rng);
        double vc = std::sqrt(G * M_central / r);  // circular speed

        b.m[i] = m_small;
        b.x[i] = r * std::cos(th);
        b.y[i] = r * std::sin(th);
        b.z[i] = 0.05 * N01(rng);  // thin disk
        // tangential velocity + a little dispersion
        b.vx[i] = -vc * std::sin(th) + 0.02 * vc * N01(rng);
        b.vy[i] =  vc * std::cos(th) + 0.02 * vc * N01(rng);
        b.vz[i] = 0.02 * vc * N01(rng);
    }

    // Shift to centre-of-mass frame so the system doesn't drift
    double px = 0, py = 0, pz = 0, mt = 0;
    for (size_t i = 0; i < b.n; ++i) {
        px += b.m[i] * b.vx[i]; py += b.m[i] * b.vy[i]; pz += b.m[i] * b.vz[i];
        mt += b.m[i];
    }
    for (size_t i = 0; i < b.n; ++i) {
        b.vx[i] -= px / mt; b.vy[i] -= py / mt; b.vz[i] -= pz / mt;
    }
}

// Each thread owns a disjoint set of i's and only writes ax[i], ay[i], az[i],
// so there are no data races. (Using Newton's 3rd law to halve the work would
// need atomics or per-thread buffers, which usually costs more than it saves.)
void compute_accel(Bodies& b) {
    const size_t n = b.n;
    const double eps2 = SOFTENING * SOFTENING;
    const double* __restrict x = b.x.get();
    const double* __restrict y = b.y.get();
    const double* __restrict z = b.z.get();
    const double* __restrict m = b.m.get();

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
            double s = G * m[j] * inv_r * inv_r * inv_r;  // j == i gives dx = 0 -> no force
            axi += s * dx;
            ayi += s * dy;
            azi += s * dz;
        }
        b.ax[i] = axi;
        b.ay[i] = ayi;
        b.az[i] = azi;
    }
}

double total_energy(const Bodies& b) {
    const size_t n = b.n;
    const double eps2 = SOFTENING * SOFTENING;
    double kin = 0.0, pot = 0.0;

#pragma omp parallel for reduction(+ : kin, pot) schedule(static)
    for (long i = 0; i < static_cast<long>(n); ++i) {
        kin += 0.5 * b.m[i] *
               (b.vx[i] * b.vx[i] + b.vy[i] * b.vy[i] + b.vz[i] * b.vz[i]);
        double p = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            double dx = b.x[j] - b.x[i];
            double dy = b.y[j] - b.y[i];
            double dz = b.z[j] - b.z[i];
            p -= G * b.m[i] * b.m[j] / std::sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
        pot += p;
    }
    return kin + pot;
}

// Kick-drift-kick leapfrog
void step(Bodies& b, double dt) {
    const long n = static_cast<long>(b.n);
    const double h = 0.5 * dt;

#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {  // half kick + drift
        b.vx[i] += b.ax[i] * h; b.vy[i] += b.ay[i] * h; b.vz[i] += b.az[i] * h;
        b.x[i] += b.vx[i] * dt; b.y[i] += b.vy[i] * dt; b.z[i] += b.vz[i] * dt;
    }

    compute_accel(b);

#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) {  // second half kick
        b.vx[i] += b.ax[i] * h; b.vy[i] += b.ay[i] * h; b.vz[i] += b.az[i] * h;
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
    size_t count;
    std::atomic<size_t> visible_count;
    HWND slider = nullptr;
    HWND pause_button = nullptr;
    HWND count_label = nullptr;
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
            view->count_label = CreateWindowExA(
                0, "STATIC", "", WS_CHILD | WS_VISIBLE | SS_RIGHT,
                0, 0, 0, 0, hwnd, reinterpret_cast<HMENU>(101),
                GetModuleHandle(nullptr), nullptr);
            view->pause_button = CreateWindowExA(
                0, "BUTTON", "Pause", WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                0, 0, 0, 0, hwnd, reinterpret_cast<HMENU>(102),
                GetModuleHandle(nullptr), nullptr);
            SendMessage(view->count_label, WM_SETFONT,
                        reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
            SendMessage(view->pause_button, WM_SETFONT,
                        reinterpret_cast<WPARAM>(GetStockObject(DEFAULT_GUI_FONT)), TRUE);
            return 0;
        }

        case WM_SIZE: {
            const int width = LOWORD(lParam);
            const int height = HIWORD(lParam);
            const int panel_height = 64;
            const int padding = 18;
            if (view != nullptr && view->slider != nullptr) {
                MoveWindow(view->slider, padding, height - 48, width - 190, 28, TRUE);
                MoveWindow(view->count_label, width - 165, height - 47, 80, 24, TRUE);
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
            const int panel_height = 64;
            const int canvas_height = std::max(1, static_cast<int>(client.bottom) - panel_height);
            const int width = std::max(1, static_cast<int>(client.right));
            const int height = std::max(1, static_cast<int>(client.bottom));

            HDC buffer = CreateCompatibleDC(dc);
            HBITMAP bitmap = CreateCompatibleBitmap(dc, width, height);
            HGDIOBJ old_bitmap = SelectObject(buffer, bitmap);
            HDC draw_dc = buffer;

            HBRUSH background = CreateSolidBrush(RGB(8, 12, 24));
            FillRect(draw_dc, &client, background);
            DeleteObject(background);

            if (view != nullptr) {
                std::lock_guard<std::mutex> lock(view->mutex);
                const double center_x = (client.right - client.left) * 0.5;
                const double center_y = canvas_height * 0.5;
                const double scale = std::min(static_cast<int>(client.right - client.left),
                                              canvas_height) / 13.0;

                HPEN grid_pen = CreatePen(PS_SOLID, 1, RGB(22, 31, 52));
                HGDIOBJ old_pen = SelectObject(draw_dc, grid_pen);
                HGDIOBJ old_brush = SelectObject(draw_dc, GetStockObject(NULL_BRUSH));
                for (int ring = 1; ring <= 5; ++ring) {
                    const int radius = static_cast<int>(ring * 1.0 * scale);
                    Ellipse(draw_dc, static_cast<int>(center_x) - radius,
                            static_cast<int>(center_y) - radius,
                            static_cast<int>(center_x) + radius,
                            static_cast<int>(center_y) + radius);
                }
                MoveToEx(draw_dc, 0, static_cast<int>(center_y), nullptr);
                LineTo(draw_dc, client.right, static_cast<int>(center_y));
                MoveToEx(draw_dc, static_cast<int>(center_x), 0, nullptr);
                LineTo(draw_dc, static_cast<int>(center_x), canvas_height);
                SelectObject(draw_dc, old_brush);
                SelectObject(draw_dc, old_pen);
                DeleteObject(grid_pen);

                HBRUSH particle_brush = CreateSolidBrush(RGB(80, 190, 255));
                HBRUSH star_brush = CreateSolidBrush(RGB(255, 220, 80));
                old_brush = SelectObject(draw_dc, particle_brush);

                const size_t visible_count = std::min(view->visible_count.load(), view->count);
                for (size_t i = 1; i < visible_count; ++i) {
                    const int px = static_cast<int>(center_x + view->x[i] * scale);
                    const int py = static_cast<int>(center_y - view->y[i] * scale);
                    Ellipse(draw_dc, px - 2, py - 2, px + 3, py + 3);
                }

                SelectObject(draw_dc, star_brush);
                const int star_radius = 7;
                Ellipse(draw_dc, static_cast<int>(center_x) - star_radius,
                        static_cast<int>(center_y) - star_radius,
                        static_cast<int>(center_x) + star_radius + 1,
                        static_cast<int>(center_y) + star_radius + 1);
                SelectObject(draw_dc, old_brush);
                DeleteObject(particle_brush);
                DeleteObject(star_brush);

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
                        std::make_unique<double[]>(bodies.n), bodies.n, bodies.n};
    publish_positions(view);

    std::thread simulation([&view, dt, threads] {
        omp_set_num_threads(threads);
        while (!view.stop) {
            const auto frame_start = std::chrono::steady_clock::now();
            while (view.paused && !view.stop) std::this_thread::sleep_for(std::chrono::milliseconds(20));
            if (view.stop) break;
            step(view.bodies, dt);
            {
                std::lock_guard<std::mutex> lock(view.mutex);
                for (size_t i = 0; i < view.count; ++i) {
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
        WS_EX_COMPOSITED, class_name, "N-body particles", WS_OVERLAPPEDWINDOW,
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