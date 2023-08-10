function quadrotor_model(N=3, backend=nothing)
    
    n = 9
    p = 4
    nd= 9
    d(i,j,N) = (j==1 ? 1*sin(2*pi/N*i) : 0.) + (j==3 ? 2*sin(4*pi/N*i) : 0.) + (j==5 ? 2*i/N : 0.)
    dt = .01
    R = fill(1/10,4)
    Q = [1,0,1,0,1,0,1,1,1]
    Qf= [1,0,1,0,1,0,1,1,1]/dt
    
    c = ExaModels.ExaCore(backend)
    
    x0s  = ExaModels.convert_array([(i,0.) for i=1:n], backend)
    itr0 = ExaModels.convert_array([(i,j,R[j]) for (i,j) in Base.product(1:N,1:p)], backend)
    itr1 = ExaModels.convert_array([(i,j,Q[j],d(i,j,N)) for (i,j) in Base.product(1:N,1:n)], backend)
    itr2 = ExaModels.convert_array([(j,Qf[j],d(N+1,j,N)) for j in 1:n], backend)

    x= ExaModels.variable(c,1:N+1,1:n)
    u= ExaModels.variable(c,1:N,1:p)
    
    ExaModels.constraint(c, x[1,i]-x0 for (i,x0) in x0s)
    ExaModels.constraint(c, -x[i+1,1] + x[i,1] + (x[i,2])*dt for i=1:N)
    ExaModels.constraint(c, -x[i+1,2] + x[i,2] + (u[i,1]*cos(x[i,7])*sin(x[i,8])*cos(x[i,9])+u[i,1]*sin(x[i,7])*sin(x[i,9]))*dt for i=1:N)
    ExaModels.constraint(c, -x[i+1,3] + x[i,3] + (x[i,4])*dt for i=1:N)
    ExaModels.constraint(c, -x[i+1,4] + x[i,4] + (u[i,1]*cos(x[i,7])*sin(x[i,8])*sin(x[i,9])-u[i,1]*sin(x[i,7])*cos(x[i,9]))*dt for i=1:N)
    ExaModels.constraint(c, -x[i+1,5] + x[i,5] + (x[i,6])*dt for i=1:N)
    ExaModels.constraint(c, -x[i+1,6] + x[i,6] + (u[i,1]*cos(x[i,7])*cos(x[i,8])-9.8)*dt for i=1:N)
    ExaModels.constraint(c, -x[i+1,7] + x[i,7] + (u[i,2]*cos(x[i,7])/cos(x[i,8])+u[i,3]*sin(x[i,7])/cos(x[i,8]))*dt for i=1:N)
    ExaModels.constraint(c, -x[i+1,8] + x[i,8] + (-u[i,2]*sin(x[i,7])+u[i,3]*cos(x[i,7]))*dt for i=1:N)
    ExaModels.constraint(c, -x[i+1,9] + x[i,9] + (u[i,2]*cos(x[i,7])*tan(x[i,8])+u[i,3]*sin(x[i,7])*tan(x[i,8])+u[i,4])*dt for i=1:N)
    ExaModels.objective(c, .5*R*(u[i,j]^2) for (i,j,R) in itr0)
    ExaModels.objective(c, .5*Q*(x[i,j]-d)^2 for (i,j,Q,d) in itr1)
    ExaModels.objective(c, .5*Qf*(x[N+1,j]-d)^2 for (j,Qf,d) in itr2)
    
    return ADBenchmarkModel(
        ExaModels.ExaModel(c)
    )
end

function jump_quadrotor_model(N= 3)
    n = 9
    p = 4
    nd= 9
    x0 = [0,0,0,0,0,0,0,0,0]
    d(i,j,N) = (j==1 ? 1*sin(2*pi/N*i) : 0) + (j==3 ? 2*sin(4*pi/N*i) : 0) + (j==5 ? 2*i/N : 0)
    dt = .01
    Q = [1,0,1,0,1,0,1,1,1]
    Qf= [1,0,1,0,1,0,1,1,1]/dt
    R = ones(4)/10

    m = JuMP.Model()
    JuMP.@variable(m,x[1:N+1,1:n],start = 0)
    JuMP.@variable(m,u[1:N,1:p],start = 0)
    JuMP.@constraint(m,[i=1:n],x[1,i]==x0[i])
    JuMP.@constraint(m,[i=1:N],x[i+1,1] == x[i,1] + (x[i,2])*dt)
    JuMP.@NLconstraint(m,[i=1:N], x[i+1,2] == x[i,2] + (u[i,1]*cos(x[i,7])*sin(x[i,8])*cos(x[i,9])+u[i,1]*sin(x[i,7])*sin(x[i,9]))*dt)
    JuMP.@constraint(m,[i=1:N], x[i+1,3] == x[i,3] + (x[i,4])*dt)
    JuMP.@NLconstraint(m,[i=1:N], x[i+1,4] == x[i,4] + (u[i,1]*cos(x[i,7])*sin(x[i,8])*sin(x[i,9])-u[i,1]*sin(x[i,7])*cos(x[i,9]))*dt)
    JuMP.@constraint(m,[i=1:N], x[i+1,5] == x[i,5] + (x[i,6])*dt)
    JuMP.@NLconstraint(m,[i=1:N], x[i+1,6] == x[i,6] + (u[i,1]*cos(x[i,7])*cos(x[i,8])-9.8)*dt)
    JuMP.@NLconstraint(m,[i=1:N], x[i+1,7] == x[i,7] + (u[i,2]*cos(x[i,7])/cos(x[i,8])+u[i,3]*sin(x[i,7])/cos(x[i,8]))*dt)
    JuMP.@NLconstraint(m,[i=1:N], x[i+1,8] == x[i,8] + (-u[i,2]*sin(x[i,7])+u[i,3]*cos(x[i,7]))*dt)
    JuMP.@NLconstraint(m,[i=1:N], x[i+1,9] == x[i,9] + (u[i,2]*cos(x[i,7])*tan(x[i,8])+u[i,3]*sin(x[i,7])*tan(x[i,8])+u[i,4])*dt)
    JuMP.@objective(m,Min, .5*sum(Q[j]*(x[i,j]-d(i,j,N))^2 for i=1:N for j=1:n) + .5*sum(R[j]*(u[i,j]^2) for i=1:N for j=1:p)
               + .5*sum(Qf[j]*(x[N+1,j]-d(N+1,j,N))^2 for j=1:n))
    return ADBenchmarkModel(
        MathOptNLPModel(m)
    )
end

function ampl_quadrotor_model(N = 3)
    nlfile = tempname()*  ".nl"
    py"""
    N = $N

    import math
    import pyomo.environ as pyo

    # Constants
    n = 9
    p = 4
    nd = 9
    x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    dt = 0.01
    Q = [1, 0, 1, 0, 1, 0, 1, 1, 1]
    Qf = [x / dt for x in Q]
    R = [0.1, 0.1, 0.1, 0.1]

    def d(i, j, N):
        if j == 1:
            return 1 * math.sin(2 * math.pi / N * i)
        elif j == 3:
            return 2 * math.sin(4 * math.pi / N * i)
        elif j == 5:
            return 2 * i / N
        else:
            return 0


    m = pyo.ConcreteModel()

    # Define the decision variables
    m.x = pyo.Var(range(1, N+2), range(1, n+1), initialize=0)
    m.u = pyo.Var(range(1, N+1), range(1, p+1), initialize=0)

    # Define the constraints
    m.constr1 = pyo.ConstraintList()
    for i in range(1, n+1):
        m.constr1.add(expr=m.x[1, i] == x0[i-1])

    m.constr2 = pyo.ConstraintList()
    for i in range(1, N+1):
        m.constr2.add(expr=m.x[i+1, 1] == m.x[i, 1] + m.x[i, 2] * dt)

    m.constr3 = pyo.ConstraintList()
    for i in range(1, N+1):
        m.constr3.add(
            expr=(
                m.x[i+1, 2] == m.x[i, 2] +
                (m.u[i, 1] * pyo.cos(m.x[i, 7]) * pyo.sin(m.x[i, 8]) * pyo.cos(m.x[i, 9]) +
                 m.u[i, 1] * pyo.sin(m.x[i, 7]) * pyo.sin(m.x[i, 9])) * dt
            )
        )

    m.constr9 = pyo.ConstraintList()
    for i in range(1, N+1):
        m.constr9.add(expr=m.x[i+1, 3] == m.x[i, 3] + m.x[i, 4] * dt)

    m.constr4 = pyo.ConstraintList()
    for i in range(1, N+1):
        m.constr4.add(
            expr=(
                m.x[i+1, 4] == m.x[i, 4] +
                (m.u[i, 1] * pyo.cos(m.x[i, 7]) * pyo.sin(m.x[i, 8]) * pyo.sin(m.x[i, 9]) -
                m.u[i, 1] * pyo.sin(m.x[i, 7]) * pyo.cos(m.x[i, 9])) * dt
            )
        )

    m.constr10 = pyo.ConstraintList()
    for i in range(1, N+1):
        m.constr10.add(expr=m.x[i+1, 5] == m.x[i, 5] + m.x[i, 6] * dt)
        
    m.constr5 = pyo.ConstraintList()
    for i in range(1, N+1):
        m.constr5.add(
            expr=(
                m.x[i+1, 6] == m.x[i, 6] +
                (m.u[i, 1] * pyo.cos(m.x[i, 7]) * pyo.cos(m.x[i, 8]) - 9.8) * dt
            )
        )

    m.constr6 = pyo.ConstraintList()
    for i in range(1, N+1):
        m.constr6.add(
            expr=(
                m.x[i+1, 7] == m.x[i, 7] +
                (m.u[i, 2] * pyo.cos(m.x[i, 7]) / pyo.cos(m.x[i, 8]) +
                m.u[i, 3] * pyo.sin(m.x[i, 7]) / pyo.cos(m.x[i, 8])) * dt
            )
        )

    m.constr7 = pyo.ConstraintList()
    for i in range(1, N+1):
        m.constr7.add(
            expr=(
                m.x[i+1, 8] == m.x[i, 8] +
                (-m.u[i, 2] * pyo.sin(m.x[i, 7]) + m.u[i, 3] * pyo.cos(m.x[i, 7])) * dt
            )
        )

    m.constr8 = pyo.ConstraintList()
    for i in range(1, N+1):
        m.constr8.add(
            expr=(
                m.x[i+1, 9] == m.x[i, 9] +
                (m.u[i, 2] * pyo.cos(m.x[i, 7]) * pyo.tan(m.x[i, 8]) +
                m.u[i, 3] * pyo.sin(m.x[i, 7]) * pyo.tan(m.x[i, 8]) + m.u[i, 4]) * dt
            )
        )

    m.obj = pyo.Objective(
        expr=(
            0.5 * sum(Q[j-1] * (m.x[i, j] - d(i, j, N))**2 for i in range(1, N+1) for j in range(1, n+1)) +
            0.5 * sum(R[j-1] * (m.u[i, j])**2 for i in range(1, N+1) for j in range(1, p+1)) +
            0.5 * sum(Qf[j-1] * (m.x[N+1, j] - d(N+1, j, N))**2 for j in range(1, n+1))
        ),
        sense=pyo.minimize
    )


    m.write($nlfile)
    """

    return ADBenchmarkModel(
        AmplNLReader.AmplModel(nlfile)
    )

end
