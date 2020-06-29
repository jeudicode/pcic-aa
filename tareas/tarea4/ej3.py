from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Creando el modelo

model = BayesianModel([
    ('ebola', 'fiebre'),
    ('ebola', 'sangrado'),
    ('fiebre', 'clinica'),
    ('sangrado', 'clinica'),
    ('sangrado', 'complicacion'),
    ('clinica', 'doctor')
])

# Se registran las probabilidades
# La clase TabularCPD  recibe el nombre de la variable, su cardinalidad y sus posibles valores
# Para el caso de las variables con dependencias, se pasan los valores de las variable padre y su cardinalidad

prob_ebola = TabularCPD("ebola", 2, [[0.01], [0.99]])
prob_fiebre = TabularCPD(
    "fiebre", 2, [[0.6, 0.1], [0.4, 0.9]], ["ebola"], [2])
prob_sangrado = TabularCPD(
    "sangrado", 2, [[0.8, 0.05], [0.2, 0.95]], ["ebola"], [2])
prob_clinica = TabularCPD("clinica", 2, [[0.8, 0.5, 0.7, 0], [
                          0.2, 0.5, 0.3, 1]], ['fiebre', 'sangrado'], [2, 2])
prob_comp = TabularCPD("complicacion", 2, [[0.75, 0.1],
                                           [0.25, 0.9]], ['sangrado'], [2])
prob_doctor = TabularCPD("doctor", 2, [[0.6, 0], [0.4, 1]], ['clinica'], [2])

model.add_cpds(prob_ebola, prob_fiebre, prob_sangrado,
               prob_clinica, prob_comp, prob_doctor)

# Obteniendo la inferencia
# Es de hacer notar que en pgmpy el valor 0 corresponde a verdadero y 1 a falso.

infer = VariableElimination(model)

res = infer.query(['ebola'], {'doctor': 0})
print(res)


# Finding Elimination Order: : 100%|██████████████████████████████████| 4/4 [00:00 < 00:00, 5312.61it/s]
# Eliminating: clinica: 100 % |█████████████████████████████████████████| 4/4 [00:00 < 00:00, 1251.66it/s]
# +----------+--------------+
# | ebola | phi(ebola) |
# += == == == == = += == == == == == == = +
# | ebola(0) | 0.0752 |
# +----------+--------------+
# | ebola(1) | 0.9248 |
# +----------+--------------+


# Resultado para el ejercicio 3.e

# res_3e_1 = infer.query(['ebola', 'clinica'], {'fiebre': 1, 'doctor': 0})
# res_3e_2 = infer.query(['ebola'], {'fiebre': 1, 'doctor': 0})
# res_3e_3 = infer.query(['clinica'], {'fiebre': 1, 'doctor': 0})
# print(res_3e_1)
# print(res_3e_2)
# print(res_3e_3)


# Finding Elimination Order: : 100%|██████████████████████████████████| 3/3 [00:00 < 00:00, 4335.94it/s]
# Eliminating: sangrado: 100 % |████████████████████████████████████████| 3/3 [00:00 < 00:00, 1236.04it/s]
# +----------+--------------+
# | ebola | phi(ebola) |
# += == == == == = += == == == == == == = +
# | ebola(0) | 0.0670 |
# +----------+--------------+
# | ebola(1) | 0.9330 |
# +----------+--------------+


# Resultado para el ejercicio extra

# res_extra = infer.query(['clinica'], {'ebola': 0})
# print(res_extra)

# Finding Elimination Order: : 100%|██████████████████████████████████| 4/4 [00:00 < 00:00, 7674.85it/s]
# Eliminating: doctor: 100 % |██████████████████████████████████████████| 4/4 [00:00 < 00:00, 2040.78it/s]
# +------------+----------------+
# | clinica | phi(clinica) |
# += == == == == == = += == == == == == == == = +
# | clinica(0) | 0.6680 |
# +------------+----------------+
# | clinica(1) | 0.3320 |
# +------------+----------------+
