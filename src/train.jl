using Flux
using DataFrames
using CSV

# Cargar y preparar los datos
data = CSV.File("path_to_data.csv")
X = Matrix(data[:, 1:end-1])  # Características
y = Matrix(data[:, end])  # Etiquetas

# Normalización de los datos (opcional)
X = (X .- mean(X, dims=1)) ./ std(X, dims=1)

# Definir el modelo
model = Chain(
    Dense(size(X, 2), 64, relu),
    Dense(64, 1)
)

# Función de pérdida
loss(x, y) = Flux.Losses.mse(model(x), y)

# Optimizador
opt = ADAM()

# Entrenamiento
for epoch in 1:100
    Flux.train!(loss, params(model), [(X, y)], opt)
    println("Epoch $epoch: Loss = $(loss(X, y))")
end

# Guardar el modelo entrenado
Flux.save("modelo_entrenado.bson", model)
