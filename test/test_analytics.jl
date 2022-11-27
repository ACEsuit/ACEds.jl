e_abs_error = matrix_entry_errors(fdata, mbfitR; mode=:abs)
mse_sum = (18*e_abs_error[:offdiag][:mse] + 12* e_abs_error[:subdiag][:mse] + 6*e_abs_error[:diag][:mse])/36
mae_sum = (18*e_abs_error[:offdiag][:mae] + 12* e_abs_error[:subdiag][:mae] + 6*e_abs_error[:diag][:mae])/36
@show e_abs_error[:all][:mse]-mse_sum
@show e_abs_error[:all][:mae]-mae_sum



diff =[]
for d in fdata
    A = reinterpret(Matrix,Matrix(Gamma(mbfitR,d.atoms)[d.friction_indices,d.friction_indices]))
    push!(diff,norm(A - transpose(A))/norm(A))
end
maximum(diff)