# Govind Chari
# Hypothesis Testing

# H0: The coin is unbiased
# Ha: The coin is biased

using Distributions

# Input Parameters
num_heads = 60
num_flips = 100
@assert num_heads <= num_flips "Number of heads must be less than or equal to the number of flips"

alpha = 0.05 # Significance level

# Mean and variance of number of heads if the coin were unbiased
mu = 0.5 * num_flips
var = 0.25 * num_flips

# Z-Score of our observation of num_heads
z = (num_heads - mu) / sqrt(var)

# Compute p-value
p = 2 * (1 - cdf(Normal(), abs(z)))

println("p-value: ", p)
if (p < alpha)
    println("p < 0.05 so we can reject the null hypothesis and conclude the coin is biased")
else
    println("p > 0.05 so we cannot reject the null hypothesis and we cannot conclude that the coin is biased")
end
