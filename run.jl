# A Toy Experiment of Constrained Monotone Neural Networks

using Lux
using Random
using Zygote
using Optimisers
using ADTypes
using CairoMakie


# Monotone Relu Layer
# ===================

struct Monotone{F1, F2} <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_weight::F1
    init_bias::F2
    isconvex::Bool
    isconcave::Bool
end

function Monotone(
    in_dims::Int, 
    out_dims::Int; 
    init_weight=Lux.glorot_uniform, 
    init_bias=Lux.zeros32,
    isconvex=false,
    isconcave=false
)
    Monotone{typeof(init_weight), typeof(init_bias)}(
        in_dims,
        out_dims, 
        init_weight, 
        init_bias, 
        isconvex, 
        isconcave
    )
end

function Lux.initialparameters(rng::AbstractRNG, m::Monotone)
    return (
        weight=m.init_weight(rng, m.out_dims, m.in_dims),
        bias=m.init_bias(rng, m.out_dims, 1)
    )
end

Lux.initialstates(::AbstractRNG, ::Monotone) = NamedTuple()
Lux.parameterlength(m::Monotone) = m.in_dims * m.out_dims + m.out_dims
Lux.statelength(::Monotone) = 0


# The activation function of the monotone ReLU layer
ρ̂(ρ˘, x) = -ρ˘(-x)
ρ̃(ρ˘, x) = ifelse.(x .< 0, ρ˘(x .+ 1) - ρ˘(x), ρ˘(x .- 1) + ρ˘(x))
function ρˢ(x, ρ˘; activaton_weights=[2, 2, 1], isconvex=false, isconcave=false) 
    isconvex && return ρ˘(x)
    isconcave && return ρ̂(ρ˘, x)

    # Split the input into convex, concave, and saturated parts
    n = size(x, 1)
    s = Int.(round.(n * activaton_weights / sum(activaton_weights)))
    s_convex = s[1]
    s_concave = s[2]
    s_saturated = n - s_convex - s_concave
    return vcat(
        ρ˘.(@view(x[1:s_convex, :])),
        ρ̂.(ρ˘, @view(x[s_convex+1:s_convex+s_saturated, :])),
        ρ̃.(ρ˘, @view(x[s_convex+s_saturated+1:end, :]))
    )
end

function (m::Monotone)(x::AbstractMatrix, ps, st::NamedTuple)
    m.out_dims == 1 && return (abs.(ps.weight) * x .+ ps.bias, st)  # Final layer
    return (ρˢ(abs.(ps.weight) * x .+ ps.bias, Lux.relu; m.isconvex, m.isconcave), st)
end


# Synthetic Data Generation
# =========================

f(x) = 5 * x + sin(4π * x)

function generate_data(rng, n; xmax=1.0, σ=0.05)
    x = xmax * rand(rng, n)
    y = f.(x) + σ * randn(rng, n)
    return x, y 
end


# Run Experiment
# ==============

function train_model(rng, model, data, opt, loss_func, n_epochs)
    tstate = Lux.Experimental.TrainState(rng, model, opt)
    vjp_rule = AutoZygote()
    loss_history = zeros(n_epochs)
    for epoch in 1:n_epochs
        _, loss, _, tstate = Lux.Experimental.single_train_step!(
            vjp_rule, loss_func, data, tstate)
        loss_history[epoch] = loss
    end
    return tstate, loss_history
end

function run()

    rng = Xoshiro(1)
    
    # parameters
    # ----------

    # Data
    N = 100
    σ = 0.1
    xmax = 1.0

    # Model
    n_epochs = 500
    ordinal_model = Chain(
        Dense(1 => 32, relu), 
        Dense(32 => 32, relu), 
        Dense(32 => 32, relu), 
        Dense(32 => 32, relu), 
        Dense(32 => 1)
    )
    monotone_model = Chain(
        Monotone(1, 32),
        Monotone(32, 32),
        Monotone(32, 32),
        Monotone(32, 32),
        Monotone(32, 1)
    )

    # ---------------- 

    # generate data
    x, y = generate_data(rng, N; σ=σ, xmax=xmax) 
    data = (x', y') .|> cpu_device()

    # train models
    ord_state, ord_loss = train_model(rng, ordinal_model, data, Adam(0.03f0), MSELoss(), n_epochs) 
    mon_state, mon_loss = train_model(rng, monotone_model, data, Adam(0.03f0), MSELoss(), n_epochs)

    # plot results
    xrange = collect(range(0, xmax, length=100))
    ord_pred = Lux.apply(ordinal_model, xrange', ord_state.parameters, ord_state.states)[1] |> vec
    mon_pred = Lux.apply(monotone_model, xrange', mon_state.parameters, mon_state.states)[1] |> vec

    fig = Figure()
    ax = Axis(fig[1, 1])
   
    scatter!(ax, x, y, color=:gray)
    lines!(ax, xrange, f.(xrange), color=:black)

    lines!(ax, xrange, ord_pred, color=:red, linestyle=:solid, label="Ordinary NN")
    lines!(ax, xrange, mon_pred, color=:blue, linestyle=:solid, label="Monotone NN")

    Legend(fig[1, 2], ax)
    
    save("result.png", fig)
end

