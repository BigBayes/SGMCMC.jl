
export void c_update_para_ispc(uniform int* uniform ux, uniform double* uniform P, uniform double* uniform g_sum_P, uniform double* uniform prec_P, uniform double* uniform inv_pr_pick, uniform double* uniform rands,
                   uniform double ka, uniform double halfstepsz, uniform double sqrtstepsz, uniform int len_ux, uniform int dim)
{
    int idx_u;
    int rnd_idx = 0;
    double inv_pr_pick_i;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        inv_pr_pick_i = inv_pr_pick[ux[i]-1];
        foreach ( k = 0 ... dim) {
            P[idx_u + k] += halfstepsz * (ka * g_sum_P[idx_u + k] - prec_P[k] * inv_pr_pick_i * P[idx_u + k]) + sqrtstepsz * rands[rnd_idx+k];
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
        rnd_idx += dim;
    }
}
