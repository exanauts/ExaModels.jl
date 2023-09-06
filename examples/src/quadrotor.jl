function quadrotor_model(N=3; T = Float64, backend=nothing, kwargs...)
    
    n = 9
    p = 4
    nd= 9
    d(i,j,N) = (j==1 ? 1*sin(2*pi/N*i) : 0.) + (j==3 ? 2*sin(4*pi/N*i) : 0.) + (j==5 ? 2*i/N : 0.)
    dt = .01
    R = fill(1/10,4)
    Q = [1,0,1,0,1,0,1,1,1]
    Qf= [1,0,1,0,1,0,1,1,1]/dt
    
    c = ExaModels.ExaCore(T, backend)
    
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
    
    return ExaModels.ExaModel(c;  kwargs...)
end

