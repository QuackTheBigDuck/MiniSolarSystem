// Parallel N-body gravity simulation (O(N^2) direct summation) using OpenMP.
//
// Build:  g++ -O3 -march=native -fopenmp -std=c++17 nbody.cpp -o nbody
// Run:    ./nbody [N=4000] [steps=200] [dt=0.001] [threads=all] [snapshot_every=0]
//
// If snapshot_every > 0, positions are written to snapshots/frame_XXXX.csv
// (create the folder first: mkdir snapshots).
//
// Units: G = 1. A heavy central body with light particles on roughly
// circular orbits in a disk (a toy "solar system / galaxy").
// Integrator: kick-drift-kick leapfrog (symplectic, good energy behaviour).

#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>

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

int main(int argc, char** argv) {
    size_t N        = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 4000;
    int steps       = argc > 2 ? std::atoi(argv[2]) : 200;
    double dt       = argc > 3 ? std::atof(argv[3]) : 0.001;
    int threads     = argc > 4 ? std::atoi(argv[4]) : omp_get_max_threads();
    int snap_every  = argc > 5 ? std::atoi(argv[5]) : 0;

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