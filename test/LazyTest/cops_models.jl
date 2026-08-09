# Lazy builders for the COPS benchmark set: each is a 1:1 transcription of the
# eager constructor in COPSBenchmark's ExaModels extension
# (COPS/ext/COPSBenchmarkExaModels/*.jl), built against args = ArgTracer() with
# the same expressions, coefficients, bounds, starts, and add_* call order, so
# materialized models must match the eager ones exactly.
# Instance scalars (h = tf/nh etc.) sit directly in expressions — no parameter
# idiom. `T`/`backend` disappear (the lazy core defers precision/backend);
# T(x) literals are baked as Float64.

cops_lazy(name, template, build) = (name = name, template = template, build = build)

const COPS_LAZY = [
    # ── bearing (COPS/ext/COPSBenchmarkExaModels/bearing.jl) ─────────────────
    cops_lazy("bearing", (nx = 1, ny = 1), args -> begin
        c = ExaCore(concrete = Val(true))
        b = 10              # grid is (0,2*pi)x(0,2*b)
        e = 0.1             # eccentricity

        hx = 2 * pi / (args.nx + 1)  # grid spacing
        hy = 2 * b / (args.ny + 1)   # grid spacing
        area = 0.5 * hx * hy         # area of triangle

        wq(i) = (1.0 + e * cos((i - 1) * hx))^3
        v0 = [max(sin((i - 1) * hx), 0.0) for i in 1:args.nx+2, j in 1:args.ny+2]

        c, v = add_var(c, 1:args.nx+2, 1:args.ny+2; lvar = 0.0, start = v0)

        c, _ = add_obj(
            c,
            0.5 * (hx * hy / 6.0) * (wq(i) + 2 * wq(i + 1)) *
            (((v[i+1, j] - v[i, j]) / hx)^2 + ((v[i, j+1] - v[i, j]) / hy)^2)
            for i in 1:args.nx+1, j in 1:args.ny+1
        )
        c, _ = add_obj(
            c,
            0.5 * (hx * hy / 6.0) * (2 * wq(i) + 2 * wq(i - 1)) *
            (((v[i-1, j] - v[i, j]) / hx)^2 + ((v[i, j-1] - v[i, j]) / hy)^2)
            for i in 2:args.nx+2, j in 2:args.ny+2
        )
        c, _ = add_obj(
            c,
            -hx * hy * e * sin((i - 1) * hx) * v[i, j] for i in 1:args.nx+2, j in 1:args.ny+2
        )

        c, _ = add_con(c, v[i, 1] for i in 1:args.nx+2)
        c, _ = add_con(c, v[i, args.ny+2] for i in 1:args.nx+2)
        c, _ = add_con(c, v[1, i] for i in 1:args.ny+2)
        c, _ = add_con(c, v[args.nx+2, i] for i in 1:args.ny+2)
        c
    end),

    # ── camshape (COPS/ext/COPSBenchmarkExaModels/camshape.jl) ───────────────
    cops_lazy("camshape", (n = 1,), args -> begin
        c = ExaCore(minimize = false, concrete = Val(true))
        R_v = 1.0         # design parameter related to the valve shape
        R_max = 2.0       # maximum allowed radius of the cam
        R_min = 1.0       # minimum allowed radius of the cam
        alpha = 1.5       # curvature limit parameter

        d_theta = 2 * pi / (5 * (args.n + 1))   # angle between discretization points

        # radius of the cam at discretization points
        c, r = add_var(c, 1:args.n; lvar = R_min, uvar = R_max, start = (R_min + R_max) / 2.0)

        c, _ = add_obj(c, (pi * R_v) / args.n * r[i] for i in 1:args.n)

        # Convexity
        c, _ = add_con(
            c,
            (-r[i-1] * r[i] - r[i] * r[i+1] + 2 * r[i-1] * r[i+1] * cos(d_theta) for i = 2:args.n-1);
            lcon = -Inf, ucon = 0.0,
        )
        c, _ = add_con(
            c,
            (-R_min * r[1] - r[1] * r[2] + 2 * R_min * r[2] * cos(d_theta) for _ in 1:1);
            lcon = -Inf, ucon = 0.0,
        )
        c, _ = add_con(
            c,
            (-R_min^2 - R_min * r[1] + 2 * R_min * r[1] * cos(d_theta) for _ in 1:1);
            lcon = -Inf, ucon = 0.0,
        )
        c, _ = add_con(
            c,
            (-r[m-1] * r[m] - r[m] * R_max + 2 * r[m-1] * R_max * cos(d_theta) for m in args.n:args.n);
            lcon = -Inf, ucon = 0.0,
        )
        c, _ = add_con(
            c,
            (-2 * R_max * r[m] + 2 * r[m]^2 * cos(d_theta) for m in args.n:args.n);
            lcon = -Inf, ucon = 0.0,
        )
        # Curvature
        c, _ = add_con(
            c,
            ((r[i+1] - r[i]) for i = 1:args.n-1);
            lcon = -alpha * d_theta, ucon = alpha * d_theta,
        )
        c, _ = add_con(
            c,
            ((r[1] - R_min) for _ in 1:1);
            lcon = -alpha * d_theta, ucon = alpha * d_theta,
        )
        c, _ = add_con(
            c,
            ((R_max - r[m]) for m in args.n:args.n);
            lcon = -alpha * d_theta, ucon = alpha * d_theta,
        )
        c
    end),

    # ── catmix (COPS/ext/COPSBenchmarkExaModels/catmix.jl) ───────────────────
    cops_lazy("catmix", (nh = 1,), args -> begin
        c = ExaCore(concrete = Val(true))
        ne = 2
        nc = 3

        h = 1 / args.nh   # Final time / nh

        rho = [
            0.11270166537926,
            0.50000000000000,
            0.88729833462074,
        ]
        bc = [1.0, 0.0]   # Boundary conditions for x
        alpha = 0.0       # Smoothing parameter
        neg_one = -1.0
        ten_T = 10.0
        one_T = 1.0
        rho_index = [(i, rho[i]) for i in 1:nc]

        # lvar/uvar/start given as zeros(T,...)/ones(T,...) in the eager
        # constructor: written as broadcast scalars here (value-identical).
        c, u = add_var(c, args.nh, nc; lvar = 0.0, uvar = 1.0, start = 0.0)
        c, v = add_var(c, args.nh, ne; start = [mod(j, ne) for i in 1:args.nh, j in 1:ne])
        c, w = add_var(c, args.nh, nc, ne; start = 0.0)
        c, pp = add_var(c, args.nh, nc, ne; start = [mod(k, ne) for i in 1:args.nh, j in 1:nc, k in 1:ne])
        c, Dpp = add_var(c, args.nh, nc, ne; start = 0.0)
        c, ppf = add_var(c, ne; start = [mod(i, ne) for i in 1:ne])

        c, _ = add_obj(c, neg_one + ppf[1] + ppf[2] for _ in 1:1)
        c, _ = add_obj(c, alpha / h * (u[i+1, j] - u[i, j])^2 for i in 1:args.nh-1, j in 1:nc)

        c, _ = add_con(
            c,
            pp[i, k, s] - v[i, s] - h * sum(w[i, j, s] * (rho^j / factorial(j)) for j in 1:nc)
            for i = 1:args.nh, (k, rho) in rho_index, s = 1:ne
        )
        c, _ = add_con(
            c,
            Dpp[i, k, s] - sum(w[i, j, s] * (rho^(j-1) / factorial(j - 1)) for j in 1:nc)
            for i = 1:args.nh, (k, rho) in rho_index, s = 1:ne
        )
        c, _ = add_con(
            c,
            ppf[s] - v[args.nh, s] - h * sum(w[args.nh, j, s] / factorial(j) for j in 1:nc) for s in 1:ne
        )
        c, _ = add_con(
            c,
            v[i, s] + sum(w[i, j, s] * h / factorial(j) for j in 1:nc) - v[i+1, s]
            for i in 1:args.nh-1, s in 1:ne
        )
        c, _ = add_con(
            c,
            Dpp[i, j, 1] - u[i, j] * (ten_T * pp[i, j, 2] - pp[i, j, 1]) for i = 1:args.nh, j = 1:nc
        )
        c, _ = add_con(
            c,
            Dpp[i, j, 2] - u[i, j] * (pp[i, j, 1] - ten_T * pp[i, j, 2]) + (one_T - u[i, j]) * pp[i, j, 2]
            for i = 1:args.nh, j = 1:nc
        )
        c, _ = add_con(
            c,
            v[1, s] - bc for (s, bc) in [(i, bc[i]) for i in 1:ne]
        )
        c
    end),

    # ── chain (COPS/ext/COPSBenchmarkExaModels/chain.jl) ─────────────────────
    cops_lazy("chain", (n = 1,), args -> begin
        c = ExaCore(concrete = Val(true))
        nh = max(2, div(args.n - 4, 4))

        L = 4
        a = 1
        b = 3
        tmin = b > a ? 1 / 4 : 3 / 4
        tf = 1.0
        h = tf / nh

        c, u = add_var(c, nh + 1; start = [4 * abs(b - a) * (k / nh - tmin) for k in 1:nh+1])
        c, x1 = add_var(c, nh + 1; start = [4 * abs(b - a) * k / nh * (1 / 2 * k / nh - tmin) + a for k in 1:nh+1])
        c, x2 = add_var(c, nh + 1; start = [(4 * abs(b - a) * k / nh * (1 / 2 * k / nh - tmin) + a) *
            (4 * abs(b - a) * (k / nh - tmin)) for k in 1:nh+1])
        c, x3 = add_var(c, nh + 1; start = [4 * abs(b - a) * (k / nh - tmin) for k in 1:nh+1])

        c, _ = add_obj(c, x2[m] for m in nh+1:nh+1)

        c, _ = add_con(c, x1[j+1] - x1[j] - 1 / 2 * h * (u[j] + u[j+1]) for j in 1:nh)
        c, _ = add_con(c, x1[1] - a for _ in 1:1)
        c, _ = add_con(c, x1[m] - b for m in nh+1:nh+1)
        c, _ = add_con(c, x2[1] for _ in 1:1)
        c, _ = add_con(c, x3[1] for _ in 1:1)
        c, _ = add_con(c, x3[m] - L for m in nh+1:nh+1)
        c, _ = add_con(c, x2[j+1] - x2[j] - 1 / 2 * h * (x1[j] * sqrt(1 + u[j]^2) + x1[j+1] * sqrt(1 + u[j+1]^2)) for j in 1:nh)
        c, _ = add_con(c, x3[j+1] - x3[j] - 1 / 2 * h * (sqrt(1 + u[j]^2) + sqrt(1 + u[j+1]^2)) for j in 1:nh)
        c
    end),

    # ── glider (COPS/ext/COPSBenchmarkExaModels/glider.jl) ───────────────────
    cops_lazy("glider", (nh = 1,), args -> begin
        c = ExaCore(concrete = Val(true))
        x_0 = 0.0
        y_0 = 1000.0
        y_f = 900.0
        vx_0 = 13.23
        vx_f = 13.23
        vy_0 = -1.288
        vy_f = -1.288
        u_c = 2.5
        r_0 = 100.0
        m = 100.0
        g = 9.81
        cd0 = 0.034
        cd1 = 0.069662
        S = 14.0
        rho = 1.13
        cL_min = 0.0
        cL_max = 1.4
        cL0 = cL_max / 2
        half = 0.5
        inv_nh = 1 / args.nh
        r_offset = 2.5
        one_T = 1.0

        c, t_f = add_var(c, 1; lvar = 0.0, start = 1.0)
        c, x = add_var(c, args.nh+1; lvar = 0.0, start = [x_0 + vx_0 * (k * inv_nh) for k in 0:args.nh])
        c, y = add_var(c, args.nh+1; start = [y_0 + (k * inv_nh) * (y_f - y_0) for k in 0:args.nh])
        c, vx = add_var(c, args.nh+1; lvar = 0.0, start = fill(vx_0, args.nh+1))
        c, vy = add_var(c, args.nh+1; start = fill(vy_0, args.nh+1))
        c, cL = add_var(c, args.nh+1; lvar = fill(cL_min, args.nh+1), uvar = fill(cL_max, args.nh+1), start = fill(cL0, args.nh+1))

        # Expressions (matching JuMP @expressions)
        c, r = add_expr(c, (x[i] / r_0 - r_offset)^2 for i in 1:args.nh+1)
        c, u = add_expr(c, u_c * (one_T - r[i]) * exp(-r[i]) for i in 1:args.nh+1)
        c, w = add_expr(c, vy[i] - u[i] for i in 1:args.nh+1)
        c, v = add_expr(c, sqrt(vx[i]^2 + w[i]^2) for i in 1:args.nh+1)
        c, D = add_expr(c, half * (cd0 + cd1 * cL[i]^2) * rho * S * v[i]^2 for i in 1:args.nh+1)
        c, L = add_expr(c, half * cL[i] * rho * S * v[i]^2 for i in 1:args.nh+1)
        c, vx_dot = add_expr(c, (-L[i] * (w[i] / v[i]) - D[i] * (vx[i] / v[i])) / m for i in 1:args.nh+1)
        c, vy_dot = add_expr(c, (L[i] * (vx[i] / v[i]) - D[i] * (w[i] / v[i])) / m - g for i in 1:args.nh+1)

        c, _ = add_obj(c, -x[k] for k in args.nh+1:args.nh+1)

        # Dynamics
        c, _ = add_con(c, x[j] - (x[j-1] + half * t_f[1] * inv_nh * (vx[j] + vx[j-1])) for j in 2:args.nh+1)
        c, _ = add_con(c, y[j] - (y[j-1] + half * t_f[1] * inv_nh * (vy[j] + vy[j-1])) for j in 2:args.nh+1)
        c, _ = add_con(c, vx[j] - (vx[j-1] + half * t_f[1] * inv_nh * (vx_dot[j] + vx_dot[j-1])) for j in 2:args.nh+1)
        c, _ = add_con(c, vy[j] - (vy[j-1] + half * t_f[1] * inv_nh * (vy_dot[j] + vy_dot[j-1])) for j in 2:args.nh+1)

        # Boundary constraints
        c, _ = add_con(c, x[1] - x_0 for _ in 1:1)
        c, _ = add_con(c, y[1] - y_0 for _ in 1:1)
        c, _ = add_con(c, y[k] - y_f for k in args.nh+1:args.nh+1)
        c, _ = add_con(c, vx[1] - vx_0 for _ in 1:1)
        c, _ = add_con(c, vx[k] - vx_f for k in args.nh+1:args.nh+1)
        c, _ = add_con(c, vy[1] - vy_0 for _ in 1:1)
        c, _ = add_con(c, vy[k] - vy_f for k in args.nh+1:args.nh+1)
        c
    end),

    # ── robot (COPS/ext/COPSBenchmarkExaModels/robot.jl) ─────────────────────
    cops_lazy("robot", (nh = 1,), args -> begin
        c = ExaCore(concrete = Val(true))
        # total length of arm
        L = 5.0
        # Upper bounds on the controls
        max_u_rho = 1.0
        max_u_the = 1.0
        max_u_phi = 1.0
        # Initial positions of the length and the angles for the robot arm
        rho0 = 4.5
        pi_T = pi
        phi0 = pi_T / 4
        half = 0.5
        third = 1 / 3
        inv_nh = 1 / args.nh
        two_pi_3 = 2 * pi_T / 3
        four_pi_3 = 4 * pi_T / 3
        zero_T = 0.0

        c, rho = add_var(c, args.nh+1; start = rho0, lvar = zero_T, uvar = L)
        c, the = add_var(c, args.nh+1; start = [two_pi_3 * (k * inv_nh)^2 for k = 1:args.nh+1], lvar = -pi_T, uvar = pi_T)
        c, phi = add_var(c, args.nh+1; start = phi0, lvar = zero_T, uvar = pi_T)
        # Derivatives
        c, rho_dot = add_var(c, args.nh+1; start = zero_T)
        c, the_dot = add_var(c, args.nh+1; start = [four_pi_3 * (k * inv_nh) for k = 1:args.nh+1])
        c, phi_dot = add_var(c, args.nh+1; start = zero_T)
        # Control
        c, u_rho = add_var(c, args.nh+1; start = zero_T, lvar = -max_u_rho, uvar = max_u_rho)
        c, u_the = add_var(c, args.nh+1; start = zero_T, lvar = -max_u_the, uvar = max_u_the)
        c, u_phi = add_var(c, args.nh+1; start = zero_T, lvar = -max_u_phi, uvar = max_u_phi)
        # Final time
        c, tf = add_var(c, 1; start = 1.0, lvar = zero_T)

        # Expressions (matching JuMP @expressions)
        c, I_the = add_expr(c, ((L - rho[i])^3 + rho[i]^3) * (sin(phi[i]))^2 * third for i = 1:args.nh+1)
        c, I_phi = add_expr(c, ((L - rho[i])^3 + rho[i]^3) * third for i = 1:args.nh+1)

        c, _ = add_obj(c, tf[1] for i = 1:1)

        # Dynamics
        c, _ = add_con(c, -rho[j] + rho[j-1] + half * tf[1] * inv_nh * (rho_dot[j] + rho_dot[j-1]) for j = 2:args.nh+1)
        c, _ = add_con(c, -phi[j] + phi[j-1] + half * tf[1] * inv_nh * (phi_dot[j] + phi_dot[j-1]) for j = 2:args.nh+1)
        c, _ = add_con(c, -the[j] + the[j-1] + half * tf[1] * inv_nh * (the_dot[j] + the_dot[j-1]) for j = 2:args.nh+1)
        c, _ = add_con(c, -rho_dot[j] + rho_dot[j-1] + half * tf[1] * inv_nh * (u_rho[j] + u_rho[j-1]) / L for j = 2:args.nh+1)
        c, _ = add_con(c, -the_dot[j] + the_dot[j-1] + half * tf[1] * inv_nh * (u_the[j] / I_the[j] + u_the[j-1] / I_the[j-1]) for j = 2:args.nh+1)
        c, _ = add_con(c, -phi_dot[j] + phi_dot[j-1] + half * tf[1] * inv_nh * (u_phi[j] / I_phi[j] + u_phi[j-1] / I_phi[j-1]) for j = 2:args.nh+1)

        # Boundary conditions
        c, _ = add_con(c, -rho[1] + rho0 for _ in 1:1)
        c, _ = add_con(c, -the[1] + zero_T for _ in 1:1)
        c, _ = add_con(c, -phi[1] + phi0 for _ in 1:1)
        c, _ = add_con(c, -rho[k] + rho0 for k in args.nh+1:args.nh+1)
        c, _ = add_con(c, -the[k] + two_pi_3 for k in args.nh+1:args.nh+1)
        c, _ = add_con(c, -phi[k] + phi0 for k in args.nh+1:args.nh+1)
        c, _ = add_con(c, -rho_dot[1] + zero_T for _ in 1:1)
        c, _ = add_con(c, -the_dot[1] + zero_T for _ in 1:1)
        c, _ = add_con(c, -phi_dot[1] + zero_T for _ in 1:1)
        c, _ = add_con(c, -rho_dot[k] + zero_T for k in args.nh+1:args.nh+1)
        c, _ = add_con(c, -the_dot[k] + zero_T for k in args.nh+1:args.nh+1)
        c, _ = add_con(c, -phi_dot[k] + zero_T for k in args.nh+1:args.nh+1)
        c
    end),

    # ── rocket (COPS/ext/COPSBenchmarkExaModels/rocket.jl) ───────────────────
    # (core bound as `core`: the eager constructor names the constant c.)
    cops_lazy("rocket", (nh = 1,), args -> begin
        core = ExaCore(minimize = false, concrete = Val(true))
        h_0 = 1.0
        v_0 = 0.0
        m_0 = 1.0
        g_0 = 1.0
        T_c = 3.5
        h_c = 500.0
        v_c = 620.0
        m_c = 0.6
        half = 0.5
        inv_nh = 1 / args.nh

        c = half * sqrt(g_0 * h_0)
        m_f = m_c * m_0
        D_c = half * v_c * (m_0 / g_0)
        T_max = T_c * m_0 * g_0

        s_v_start = [i * inv_nh * (1.0 - i * inv_nh) for i = 0:args.nh]
        s_m_start = [(m_f - m_0) * (i * inv_nh) + m_0 for i = 0:args.nh]
        s_Th_start = T_max * half
        core, h = add_var(core, 0:args.nh; start = h_0, lvar = h_0)
        core, v = add_var(core, 0:args.nh; start = s_v_start, lvar = v_0)
        core, m = add_var(core, 0:args.nh; start = s_m_start, lvar = m_f, uvar = m_0)
        core, Th = add_var(core, 0:args.nh; start = s_Th_start, lvar = v_0, uvar = T_max)
        core, step = add_var(core, 1; start = inv_nh, lvar = v_0)

        core, _ = add_obj(core, h[k] for k in args.nh:args.nh)

        # Dynamics
        core, _ = add_con(core, -h[i] + h[i-1] + half * step[1] * (v[i] + v[i-1]) for i = 1:args.nh)
        core, _ = add_con(core, -v[i] + v[i-1] + half * step[1] * ((Th[i] - D_c * v[i]^2 * exp(-h_c * (h[i] - h_0)) / h_0 - m[i] * g_0 * (h_0 / h[i])^2) / m[i] + (Th[i-1] - D_c * v[i-1]^2 * exp(-h_c * (h[i-1] - h_0)) / h_0 - m[i-1] * g_0 * (h_0 / h[i-1])^2) / m[i-1]) for i = 1:args.nh)
        core, _ = add_con(core, -m[i] + m[i-1] + half * step[1] * (-Th[i] / c + -Th[i-1] / c) for i = 1:args.nh)

        # Boundary constraints
        core, _ = add_con(core, h[0] - h_0 for _ in 1:1)
        core, _ = add_con(core, v[0] - v_0 for _ in 1:1)
        core, _ = add_con(core, m[0] - m_0 for _ in 1:1)
        core, _ = add_con(core, m[k] - m_f for k in args.nh:args.nh)
        core
    end),

    # ── steering (COPS/ext/COPSBenchmarkExaModels/steering.jl) ───────────────
    cops_lazy("steering", (nh = 1,), args -> begin
        c = ExaCore(concrete = Val(true))
        a = 100.0  # Magnitude of force.
        # Bounds on the control
        u_min, u_max = -pi / 2, pi / 2
        xs = zeros(4)
        xf = [NaN, 5.0, 45.0, 0.0]
        half = 0.5
        inv_nh = 1 / args.nh

        function gen_x0(k, i)
            if i == 1 || i == 4
                return 0.0
            elseif i == 2
                return 5.0 * k * inv_nh
            elseif i == 3
                return 45.0 * k * inv_nh
            else
                return 0.0
            end
        end

        c, u = add_var(c, 1:args.nh+1; lvar = u_min, uvar = u_max, start = 0.0)   # control
        c, x = add_var(c, 1:args.nh+1, 1:4; start = [gen_x0(i, j) for i = 1:args.nh+1, j = 1:4])  # state
        c, tf = add_var(c, 1; start = 1.0)                                        # final time

        c, _ = add_obj(c, tf[1] for _ in 1:1)

        c, _ = add_con(c, (tf[1] for _ in 1:1); lcon = 0.0, ucon = Inf)
        # Dynamics
        c, _ = add_con(c, -x[i+1, 1] + x[i, 1] + half * (tf[1] * inv_nh) * (x[i, 3] + x[i+1, 3]) for i = 1:args.nh)
        c, _ = add_con(c, -x[i+1, 2] + x[i, 2] + half * (tf[1] * inv_nh) * (x[i, 4] + x[i+1, 4]) for i = 1:args.nh)
        c, _ = add_con(c, -x[i+1, 3] + x[i, 3] + half * (tf[1] * inv_nh) * (a * cos(u[i]) + a * cos(u[i+1])) for i = 1:args.nh)
        c, _ = add_con(c, -x[i+1, 4] + x[i, 4] + half * (tf[1] * inv_nh) * (a * sin(u[i]) + a * sin(u[i+1])) for i = 1:args.nh)
        # Boundary conditions
        c, _ = add_con(c, -x[1, j] + s for (j, s) in enumerate(xs))
        c, _ = add_con(c, -x[args.nh+1, j] + f for (j, f) in zip(2:4, xf[2:4]))
        c
    end),

    # ── torsion (COPS/ext/COPSBenchmarkExaModels/torsion.jl) ─────────────────
    cops_lazy("torsion", (nx = 1, ny = 1), args -> begin
        c = ExaCore(concrete = Val(true))
        c_val = 5.0
        hx = 1.0 / (args.nx + 1.0)
        hy = 1.0 / (args.ny + 1.0)
        area = 0.5 * hx * hy

        # Distance to boundary: D[k1,k2] for k1 in 1:nx+2, k2 in 1:ny+2
        D = [min(min(i, args.nx - i + 1) * hx, min(j, args.ny - j + 1) * hy) for i in 0:args.nx+1, j in 0:args.ny+1]

        c, v = add_var(c, args.nx+2, args.ny+2; start = D)

        # Objective = area * ((quadLower + quadUpper)/2 - c*(linLower + linUpper)/3)
        c, _ = add_obj(c, area / 2 * (((v[k1+1, k2] - v[k1, k2]) / hx)^2 + ((v[k1, k2+1] - v[k1, k2]) / hy)^2)
            for k1 in 1:args.nx+1, k2 in 1:args.ny+1)
        c, _ = add_obj(c, area / 2 * (((v[k1, k2] - v[k1-1, k2]) / hx)^2 + ((v[k1, k2] - v[k1, k2-1]) / hy)^2)
            for k1 in 2:args.nx+2, k2 in 2:args.ny+2)
        c, _ = add_obj(c, -area * c_val / 3 * (v[k1+1, k2] + v[k1, k2] + v[k1, k2+1])
            for k1 in 1:args.nx+1, k2 in 1:args.ny+1)
        c, _ = add_obj(c, -area * c_val / 3 * (v[k1, k2] + v[k1-1, k2] + v[k1, k2-1])
            for k1 in 2:args.nx+2, k2 in 2:args.ny+2)

        # Bound constraints on v: -D <= v <= D (matching JuMP's @constraint formulation)
        D_flat = [(k1, k2, D[k1, k2]) for k1 in 1:args.nx+2, k2 in 1:args.ny+2]
        c, _ = add_con(c, (v[k1, k2] for (k1, k2, d) in D_flat);
            lcon = [-d for (_, _, d) in D_flat], ucon = [d for (_, _, d) in D_flat])
        c
    end),
]

# Models whose eager constructors bake instantiation-argument values into the
# model structure (index data, loop ranges, branches) or carry no size argument
# at all — not expressible as lazy builders without guessing new vocabulary.
const COPS_UNSUPPORTED = [
    (name = "channel", reason = "start matrix v0 filled by a loop over the size-dependent range 1:nh (and uc0/coefficient tables derived from it) at channel.jl:21-27 — structure-on-args"),
    (name = "elec", reason = "nested pair comprehension [(i,j) for i in 1:np-1 for j in i+1:np] — inner range depends on the size parameter — at elec.jl:21 (also seeded rand(np) start data at elec.jl:12-13)"),
    (name = "gasoil", reason = "measurement partition itau = Int[min(nh, Int(floor(tau[i]/h))+1) ...] freezes size-dependent values into variable indices and v0-loop ranges at gasoil.jl:27,56-64,77 — structure-on-args"),
    (name = "marine", reason = "measurement partition itau (Int(floor(tau[i]/h))) drives v0-loop ranges and objective iterator indices at marine.jl:21,50-58,77 — structure-on-args"),
    (name = "methanol", reason = "measurement partition itau (Int(floor(tau[i]/h))) indexes t[itau[j]] in the objective iterator con1_matrix at methanol.jl:42,64 — structure-on-args"),
    (name = "minsurf", reason = "start surface v0 built by a mutating loop over size-dependent ranges from a LinRange mesh at minsurf.jl:13-18 (also Int(floor(0.25/hx)) generator bounds at minsurf.jl:67) — structure-on-args"),
    (name = "pinene", reason = "measurement partition itau (Int(floor(tau[i]/h))) drives v0-loop ranges and objective iterator indices at pinene.jl:26,42-50,62 — structure-on-args"),
    (name = "polygon", reason = "nested pair comprehension [(i, j) for i in 1:N-1 for j in i+1:N] — inner range depends on the size parameter — at polygon.jl:28"),
    (name = "transition_state", reason = "fixed-size instance (no size argument to defer): takes (problem, dom::PDEDiscretizationDomain) data structures at pde_models.jl:2"),
    (name = "dirichlet", reason = "start point x0 = _initial_position!(...) at pde_models.jl:10 iterates and branches on nh-dependent energies (pde_utils.jl:107-146); PDEDiscretizationDomain freezes BREAK::Int = nh (pde_utils.jl:41-48) — structure-on-args"),
    (name = "henon", reason = "start point x0 = _initial_position!(...) at pde_models.jl:10 iterates and branches on nh-dependent energies (pde_utils.jl:107-146); PDEDiscretizationDomain freezes BREAK::Int = nh (pde_utils.jl:41-48) — structure-on-args"),
    (name = "lane_emden", reason = "start point x0 = _initial_position!(...) at pde_models.jl:10 iterates and branches on nh-dependent energies (pde_utils.jl:107-146); PDEDiscretizationDomain freezes BREAK::Int = nh (pde_utils.jl:41-48) — structure-on-args"),
    (name = "tetra_duct12", reason = "fixed-size instance (no size argument to defer) at tetra.jl:65"),
    (name = "tetra_duct15", reason = "fixed-size instance (no size argument to defer) at tetra.jl:66"),
    (name = "tetra_duct20", reason = "fixed-size instance (no size argument to defer) at tetra.jl:67"),
    (name = "tetra_hook", reason = "fixed-size instance (no size argument to defer) at tetra.jl:68"),
    (name = "tetra_foam5", reason = "fixed-size instance (no size argument to defer) at tetra.jl:69"),
    (name = "tetra_gear", reason = "fixed-size instance (no size argument to defer) at tetra.jl:70"),
    (name = "triangle_deer", reason = "fixed-size instance (no size argument to defer) at triangle.jl:52"),
    (name = "triangle_pacman", reason = "fixed-size instance (no size argument to defer) at triangle.jl:53"),
    (name = "triangle_turtle", reason = "fixed-size instance (no size argument to defer) at triangle.jl:54"),
]

# transcribed: 9, unsupported: 21
