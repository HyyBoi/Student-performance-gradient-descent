# Predicting Student Grades with Gradient Descent

A small machine-learning project built from scratch for my Multivariable Calculus course (MT 1008). The idea: a student's final grade depends on several things at once (assignments, quizzes, and two sessional exams), so instead of looking at each one separately, I model them together as a function of four variables and let **gradient descent** find the best weights.

No scikit-learn, no NumPy. Just plain Python, so every derivative and every update step is written out and visible.

## The model

```
F(x1, x2, x3, x4) = w1·x1 + w2·x2 + w3·x3 + w4·x4 + b
```

| Variable | Meaning |
|---|---|
| x1 | Assignments score |
| x2 | Quiz score |
| x3 | Sessional I score |
| x4 | Sessional II score |
| w1…w4 | How much each component contributes |
| b | Bias (baseline performance) |

**Cost function** (mean squared error):

```
J(w, b) = (1/m) · Σ (F(xᵢ) − yᵢ)²
```

**Gradient** (one partial derivative per parameter):

```
∂J/∂wⱼ = (2/m) · Σ (F(xᵢ) − yᵢ) · xⱼᵢ
∂J/∂b  = (2/m) · Σ (F(xᵢ) − yᵢ)
```

Each iteration nudges every parameter a small step *against* its partial derivative, which is steepest descent on the error surface.

## What I did

1. Loaded the training data (Fall 2025 class) and picked the four inputs.
2. Built the linear model and the MSE cost function.
3. Derived the partial derivatives by hand and coded them.
4. Ran gradient descent for 5000 iterations with three learning rates.
5. Took the best-converging weights and predicted final marks for an unseen class (Spring 2026, 57 students).
6. Classified each student as pass/fail using a 50/100 cut-off.
7. Interpreted the weights and gradients, and compared the model to a single artificial neuron.

## Results

**Learning rate matters a lot.** Same data, same 5000 iterations:

| α | Outcome |
|---|---|
| 0.001 | Converged. Cost fell from ~1194 to ~18.01 |
| 0.01 | Diverged (cost → NaN) |
| 0.1 | Diverged (cost → NaN) |

**Trained parameters (α = 0.001):**

| Parameter | Value |
|---|---|
| w1 (Assignments) | 2.5122 |
| w2 (Quizzes) | 1.5326 |
| w3 (Sessional I) | 1.9473 |
| w4 (Sessional II) | 2.0919 |
| b (bias) | 0.6829 |

Final training MSE ≈ 18.01, which is an RMSE of roughly 4.2 marks out of 100. The gradient at the solution is essentially zero (∂J/∂w1 ≈ 0.007), which is what you'd expect at a minimum. Running the script prints the full gradient vector for your own data.

**Predictions on the unseen class:** 42 of 57 students predicted to pass, 15 to fail. The per-student table comes from running the script on the course data (not included, see below).

In the report we concluded that assignments had the strongest influence, since they got the largest weight.

## Things I'd be upfront about

I'd rather list these than have someone else find them:

- **No held-out accuracy.** The unseen class had no final grades attached, so I can't report test error. The 4.2 RMSE is on the training data only.
- **Raw weights aren't a perfect "importance" ranking.** The inputs have different score ranges, so a bigger weight doesn't automatically mean a bigger effect. Standardising the features would make the comparison fairer.
- **Feature scaling would also fix the divergence.** α = 0.01 blows up mainly because the inputs are unscaled and the cost surface is steeply curved in some directions.
- **Missing marks were filled with 0**, which probably underestimates those students (they're flagged in the output).
- **The pass mark of 50 is an assumption**, and the model is linear and ignores anything non-academic.

## Running it

You need Python 3.8+.

**Quick demo (no data needed).** Runs on synthetic students so you can see the whole pipeline:

```bash
python student_performance_model.py --demo
```

**With real data.** The course spreadsheets contain real students' marks, so they are **not** included in this repo. If you have files with the same layout:

```bash
pip install -r requirements.txt
python student_performance_model.py --train Dataset_1.xlsx --test Dataset_2.xlsx
```

The column positions used to read the spreadsheets are hard-coded in `load_training_data` and `load_test_data`, so you'd need to adjust them for a different layout.

## Repo layout

```
.
├── student_performance_model.py   # model, cost, gradients, gradient descent, prediction
├── requirements.txt               # only needed for reading .xlsx files
├── LICENSE
└── README.md
```

## What I'd do next

- Standardise the inputs and re-run, so all learning rates behave and weights are comparable.
- Hold out part of the training class to get a real test error.
- Compare against the closed-form least-squares solution as a sanity check.
- Try a learning-rate schedule instead of a fixed α.

## Course context

Made for MT 1008 Multivariable Calculus. Concepts used: partial derivatives, gradient vectors, directional change, steepest descent, and optimisation of a multivariable function.

## Authors

Abdul Hannan Muhammad ([@HyyBoi](https://github.com/HyyBoi))