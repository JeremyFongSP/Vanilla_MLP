using Printf
using CSV
using DataFrames
using Random
using Plots
theme(:juno)
using Statistics

# FOR PLOTTING
gr(reuse = true)

# SET PATHS AND FILE NAME
dir = pwd()
file_train = joinpath(dir, "../Data/OCT_Made_up_data.csv")
file_test = joinpath(dir, "../Data/OCT_Made_up_test.csv")
println("Working in the following directory:")
println(dir)
println("Using Training File:")
println(file_train)
println("Using Testing File:")
println(file_test)

ϵ = 1e-9

# LOAD DATA
df1 = CSV.read(file_train, header=false)
y = df1[!, end]
X = delete!(df1, size(df1)[2])
X_array = convert(Matrix, X)
N_DATA = size(X)[1]

# NORMALIZING -> X - μ / √(σ²)
# ϵ prevents dividing by 0
μ = mean(X_array, dims=1)
σ² = var(X_array, dims=1)
X_array = (X_array .- μ) ./ sqrt.(σ² .+ ϵ)

# LAYERS
NIN = 400
H1 = 256
H2 = 256
NOUT = 10

# HYPERPARAMETERS
# The alpha has to be quite small in order for it to
# converge at all. At a value of 0.3, the Neural Net
# isn't able to learn anything valuable, even after
# 1000 epochs
λ = 0
α = 1e-5

# SETTINGS
EPOCHS = 200
BATCH_SIZE = 100
N_BATCH = ceil(Int, N_DATA/BATCH_SIZE)
DROPOUT = 0.5
ACC_THRESHOLD = 0.98

# METHODS
sigmoid(z) = 1.0 ./ (1.0 .+ exp.(-z))
sigmoidgradient(z) = z .* (1.0.-z)
softmax(z) = exp.(z) ./ (sum(exp.(z)).+ϵ)
drop(n) = DROPOUT .<= rand(n)

# START PROGRAM
println("--Neural Network--")

# --- TRAIN NEURAL NETWORK --- #

# INIT
θi1 = 0.15*randn(NIN+1, H1)
θ12 = 0.15*randn(H1+1, H2)
θ2o = 0.15*randn(H2+1,NOUT)

# BATCH ERROR
δθi1 = zeros(NIN+1, H1)
δθ12 = zeros(H1+1, H2)
δθ2o = zeros(H2+1, NOUT)

# PLOTTING
p = plot([0],[0], linewidth=2,
         title="Correct/Total Train", color="cyan",
         xlabel="10's Batch trained", ylabel="Percent",
         legend=:bottomright)
gui()

for l = 1:EPOCHS
    global α
    println("EPOCH $l:")
    ROW_ARRAY = collect(1:N_DATA)
    shuffle!(ROW_ARRAY)

    for i = 1:N_BATCH
        # Must explicitly say that these variables aren't local
        # so that their values can be modified
        global θi1, δθi1
        global θ12, δθ12
        global θ2o, δθ2o

        correct = 0.0
        total = 0.0

        # If there's not enough examples to fill a batch size,
        # use all of the remaining examples
        if size(ROW_ARRAY,1) < BATCH_SIZE
            batch_size = size(ROW_ARRAY,1)
        else
            batch_size = BATCH_SIZE
        end

        # DROPOUT MASK
        # Every iterations, different nodes are being dropped
        # so that when gradient descent happens, it modifies
        # the weight of different neurons rather than perhaps
        # being stuck with the highest weighted one during the
        # entire training process
        r1 = drop(H1)
        r2 = drop(H2)

        for j = 1:batch_size
            # SELECT EXAMPLE
            k = pop!(ROW_ARRAY)
            x = X_array[k,:]

            # ANSWER KEY
            y_actual = zeros(NOUT)
            y_actual[y[k]+1] = 1.0

            # FORWARD PROPAGATION
            a0 = x
            push!(a0,1)
            z1 = θi1'*a0
            a1 = sigmoid(z1) .* r1
            push!(a1,1)
            z2 = θ12'*a1
            a2 = sigmoid(z2) .* r2
            push!(a2,1)
            z3 = θ2o'*a2
            a3 = sigmoid(z3)
            y_predict = a3

            # BACKPROPAGATION
            # 1:end-1 removes the bias term
            δ3 = y_predict - y_actual
            δ2 = (θ2o*δ3) .* sigmoidgradient(a2)
            δ1 = (θ12*δ2[1:end-1]) .* sigmoidgradient(a1)

            # CUMULATIVE ERROR
            δθi1 += a0*δ1[1:end-1]'
            δθ12 += a1*δ2[1:end-1]'
            δθ2o += a2*δ3'

            # PREDICTION & COUNT
            prediction = findmax(y_predict)[2]-1
            actual = y[k]
            if prediction == actual
                correct += 1.0
            end
            total += 1.0
        end

        # GRADIENT DESCENT
        θi1 -= α*δθi1
        θ12 -= α*δθ12
        θ2o -= α*δθ2o

        # DISPLAY PROGRESS (every 10 batches or 1000 examples)
        if i%10 == 0
            result = correct/total
            push!(p,4*(l-1)+i/10,result)
            gui()
            display(p)
            println("% Correct:", round(result*100.0, digits=2))
            if result >= ACC_THRESHOLD
                println("Reached $(ACC_THRESHOLD*100.0)% threshold")
                return
            end
        end
    end
    # update the steps we take after different epochs
    # smaller steps when we are close to the minimum
    # Starts by multiplying by 99/100, then 98/99, 97/98 etc
    α *= (EPOCHS-l)/(EPOCHS-l+1)
end

# SCALE WEIGHTS BY DROPOUT FACTOR
θi1 *= 1.0-DROPOUT
θ12 *= 1.0-DROPOUT
θ2o *= 1.0-DROPOUT

# --- TEST NEURAL NETWORK --- #
println("Testing against new data...")

# LOAD DATA
df2 = CSV.read(file_test, header=false)
y_test = df2[!, end]
X_test = delete!(df2, size(df2)[2])
X_array_test = convert(Matrix, X_test)
N_DATA_test = size(X_test)[1]
ROW_ARRAY_test = collect(1:N_DATA_test)

# NORMALIZE
X_array_test = (X_array_test .- μ) ./ sqrt.(σ² .+ ϵ)

# INIT
correct_test = 0.0
total_test = 0.0
history = []

for j = 1:N_DATA_test
    global correct_test, total_test

    # SELECT EXAMPLE (in order this time)
    k = pop!(ROW_ARRAY_test)
    x = X_array_test[k,:]

    # ANSWER KEY
    y_actual = zeros(NOUT)
    y_actual[y_test[k]+1] = 1.0

    # FORWARD PROPAGATION
    a0 = x
    push!(a0,1)
    z1 = θi1'*a0
    a1 = sigmoid(z1)
    push!(a1,1)
    z2 = θ12'*a1
    a2 = sigmoid(z2)
    push!(a2,1)
    z3 = θ2o'*a2
    a3 = sigmoid(z3)
    y_predict = a3

    # PREDICTION & COUNT
    prediction_test = findmax(y_predict)[2]-1
    append!(history, prediction_test)
    actual_test = y_test[k]
    if prediction_test == actual_test
        correct_test += 1.0
    end
    total_test += 1.0
end

# TEST RESULT
result_test = round((correct_test/total_test)*100.0)
println("Correct: ", correct_test)
println("Total: ", total_test)
println("Test Accuracy: $result_test%")
println("Predictions:")
println(history)
