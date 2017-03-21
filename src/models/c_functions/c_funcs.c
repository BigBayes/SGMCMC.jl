#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "./c_matrix.h"
#include "./c_funcs_ispc.h"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 *double drand()   [> uniform distribution, (0..1] <]
 *{
 *    return (rand()+1.0)/(RAND_MAX+1.0);
 *}
 *
 *double random_normal()  [> normal distribution, centered on 0, std dev 1 <]
 *{
 *    return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
 *}
 *
 */

double dot_product(double* a, double* b, int n)
{
    double result = 0.0;
    for (int i = 0; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}


//@time R_AB = R - A[uu] - B[ii]
//@time c_comp_R_AB_bpmf!(R_AB, R, A, B, uu, ii);
void c_comp_R_AB_bpmf(double* R_AB, double* R, double* A, double* B, int* uu, int* ii, int N)
{
    for (int i=0; i < N; i++)
        R_AB[i] = R[i] - A[uu[i]-1] - B[ii[i]-1];
}

//#@time XX = C - B[ii]
void c_comp_XX_bpmf(double* XX, double* C, double* B, int* ii, int N)
{
    for (int i=0; i < N; i++)
        XX[i] = C[i] - B[ii[i]-1];
}

void c_comp_C_bpmf(double* C, double* R, double* P, double* Q,
                   int* uu, int* ii, int N, int D)
{
    int idx_u, idx_i;
    for (int i = 0; i < N; i++)
    {
        idx_u = D * (uu[i] - 1);
        idx_i = D * (ii[i] - 1);
        C[i] = R[i] - dot_product(&P[idx_u], &Q[idx_i], D);
    }
}

void c_comp_error(double* error, double* rr, double mean_rate, double* P, double* Q, double* A, double* B,
                  int* uu, int* ii, int size, int D)
{
    int idx_u, idx_i;
    /*double dotval;*/
    for (int i = 0; i < size; i++)
    {
        idx_u = D * (uu[i] - 1);
        idx_i = D * (ii[i] - 1);
        error[i] = (rr[i] - mean_rate) - (dot_product(&P[idx_u], &Q[idx_i], D) + A[uu[i]-1] + B[ii[i] - 1]);
    }
}

void c_comp_error_m1(double* error, double* rr, double mean_rate, double* P, double* Q,
                  int* uu, int* ii, int size, int D)
{
    int idx_u, idx_i;
    /*double dotval;*/
    for (int i = 0; i < size; i++)
    {
        idx_u = D * (uu[i] - 1);
        idx_i = D * (ii[i] - 1);
        error[i] = (rr[i] - mean_rate) - dot_product(&P[idx_u], &Q[idx_i], D);
    }
}


void c_comp_grad_sum(double* error, int* uu, int* ii,
                     double* P, double* Q, double* A, double* B,
                     double* g_sum_P, double* g_sum_Q,
                     double* g_sum_A, double* g_sum_B, int m, int D)
{
    int uu_i, ii_i, uu_i_d_idx, ii_i_d_idx;
    double err_i;
    for (int i = 0; i < m; i++) {
        uu_i  = uu[i] - 1;
        ii_i  = ii[i] - 1;
        err_i = error[i];
        for (int d = 0; d < D; d++) {
            uu_i_d_idx = D * uu_i + d;
            ii_i_d_idx = D * ii_i + d;
            g_sum_P[uu_i_d_idx] = g_sum_P[uu_i_d_idx] + (err_i * Q[ii_i_d_idx]);
            g_sum_Q[ii_i_d_idx] = g_sum_Q[ii_i_d_idx] + (err_i * P[uu_i_d_idx]);
        }
        g_sum_A[uu_i] += err_i;
        g_sum_B[ii_i] += err_i;
    }
}

void c_comp_grad_sum_m1(double* error, int* uu, int* ii, double* P, double* Q, double* g_sum_P, double* g_sum_Q, int m, int D)
{
    int uu_i, ii_i, uu_i_d_idx, ii_i_d_idx;
    double err_i;
    for (int i = 0; i < m; i++) {
        uu_i  = uu[i] - 1;
        ii_i  = ii[i] - 1;
        err_i = error[i];
        for (int d = 0; d < D; d++) {
            uu_i_d_idx = D * uu_i + d;
            ii_i_d_idx = D * ii_i + d;
            g_sum_P[uu_i_d_idx] = g_sum_P[uu_i_d_idx] + (err_i * Q[ii_i_d_idx]);
            g_sum_Q[ii_i_d_idx] = g_sum_Q[ii_i_d_idx] + (err_i * P[uu_i_d_idx]);
        }
    }
}


/*
 *void c_comp_grad_sum_pmf_m1(double* error, int* uu, int* ii, double* P, double* Q, double* g_sum_P, double* g_sum_Q, int m, int D)
 *{
 *    int uu_i, ii_i, uu_i_d_idx, ii_i_d_idx;
 *    double err_i;
 *    for (int i = 0; i < m; i++) {
 *        uu_i  = uu[i] - 1;
 *        ii_i  = ii[i] - 1;
 *        err_i = error[i];
 *        for (int d = 0; d < D; d++) {
 *            uu_i_d_idx = D * uu_i + d;
 *            ii_i_d_idx = D * ii_i + d;
 *            g_sum_P[uu_i_d_idx] = g_sum_P[uu_i_d_idx] + (err_i * Q[ii_i_d_idx]);
 *            g_sum_Q[ii_i_d_idx] = g_sum_Q[ii_i_d_idx] + (err_i * P[uu_i_d_idx]);
 *        }
 *    }
 *}
 */

void c_update_para(int* ux, double* P, double* g_sum_P, double* prec_P, double* inv_pr_pick, double* rands,
                   double ka, double halfstepsz, double sqrtstepsz, int len_ux, int dim)
{
    int idx_u;
    int rnd_idx = 0;
    double inv_pr_pick_i;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        inv_pr_pick_i = inv_pr_pick[ux[i]-1];
        for (int k = 0; k < dim; k++) {
            P[idx_u + k] += halfstepsz * (ka * g_sum_P[idx_u + k] - prec_P[k] * inv_pr_pick_i * P[idx_u + k]) + sqrtstepsz * rands[rnd_idx++];
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}

void c_update_para_prior(int* ux, double*restrict P, double*restrict g_sum_P, double*restrict prec_P, double*restrict prior_muprec, double*restrict prior_negprec, double*restrict inv_pr_pick, double*restrict rands,
                   double ka, double halfstepsz, double sqrtstepsz, int len_ux, int dim)
{
    int idx_u;
    int rnd_idx = 0;
    double inv_pr_pick_i;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        inv_pr_pick_i = inv_pr_pick[ux[i]-1];
        for (int k = 0; k < dim; k++) {
            P[idx_u + k] += halfstepsz * (ka * g_sum_P[idx_u + k] - inv_pr_pick_i * (( prec_P[k] - prior_negprec[idx_u + k]) *  P[idx_u + k] - prior_muprec[idx_u + k])  ) + sqrtstepsz * rands[rnd_idx++];
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}

void c_update_para_prior_sghmc(int* ux, double*restrict P, double*restrict mom_P, double*restrict g_sum_P, double*restrict prec_P, double*restrict prior_muprec, double*restrict prior_negprec, double*restrict rands,
                   double ka, double stepsz, double sqrtstepsz, double A, int len_ux, int dim)
{
    int idx_u;
    int rnd_idx = 0;
    double prec;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        for (int k = 0; k < dim; k++) {
            prec =   prec_P[k] - prior_negprec[idx_u + k];
            mom_P[idx_u + k] += stepsz * (ka * g_sum_P[idx_u + k] - A * mom_P[idx_u +k] - ( prec*(P[idx_u + k] - prior_muprec[idx_u + k]/prec)  )) + sqrtstepsz * rands[rnd_idx++];
            P[idx_u + k] += stepsz * mom_P[idx_u + k ];
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}

void c_update_para_prior_sgnht(int*restrict ux, double*restrict P, double*restrict mom_P, double*restrict g_sum_P, double*restrict prec_P, double*restrict prior_muprec, double*restrict prior_negprec, double*restrict rands,  double ka, double stepsz, double sqrtstepsz, double xi, int len_ux, int dim)
{
    int idx_u;
    int rnd_idx = 0;
    double prec;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        for (int k = 0; k < dim; k++) {
            prec =   prec_P[k] - prior_negprec[idx_u + k];
            mom_P[idx_u + k] += stepsz * (ka * g_sum_P[idx_u + k] - xi* mom_P[idx_u +k] - ( prec*(P[idx_u + k] - prior_muprec[idx_u + k]/prec)  )) + sqrtstepsz * rands[rnd_idx++];
            P[idx_u + k] += stepsz * mom_P[idx_u + k ];
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}


void c_update_para_prior_msgnht(int*restrict ux, double*restrict P, double*restrict mom_P, double* restrict xi_P, double*restrict g_sum_P, double*restrict prec_P, double*restrict prior_muprec, double*restrict prior_negprec, double*restrict rands,  double ka, double stepsz, double sqrtstepsz, double mu, int len_ux, int dim)
{
    int idx_u;
    int rnd_idx = 0;
    double prec;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        for (int k = 0; k < dim; k++) {
            prec =   prec_P[k] - prior_negprec[idx_u + k];
            xi_P[idx_u + k] += stepsz/mu*(mom_P[idx_u +k ]*mom_P[idx_u + k] - 1.);
            mom_P[idx_u + k] += stepsz * (ka * g_sum_P[idx_u + k] - xi_P[idx_u+k]* mom_P[idx_u +k] - ( prec*(P[idx_u + k] - prior_muprec[idx_u + k]/prec)  )) + sqrtstepsz * rands[rnd_idx++];
            P[idx_u + k] += stepsz * mom_P[idx_u + k ];
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}


double* get_SSHM_covariance_pos(double omega, double gamma, double beta , double cos_2t, double sin_2t, double e_gammat)
{
    double gamma2 = gamma*gamma;
    double omegar2 = omega*omega - gamma2/4;
    double beta_2 = beta*beta;
    double *C = (double*)calloc(2 * 2, sizeof(double));
    double omega2 = omega *omega;
    double omegar = sqrt(omegar2);
    C[0] = 1.0/(omega2) + e_gammat/(4*omega2* omegar2)*(-4*omega2+gamma2*cos_2t-2*gamma*omegar*sin_2t);
    C[1] =  e_gammat*beta_2/(4*omegar2)*(1.0-cos_2t);
    C[2] = C[1];
    C[3] = 1.+ e_gammat/(4*omegar2)*(-4*omega2+gamma2*cos_2t+2*gamma*omegar*sin_2t);
    return C;
}

double* get_SSHM_covariance_neg(double omega, double gamma, double beta , double cosh_2t, double sinh_2t)
{
    //printf("%f %f %f %f %f\n", omega, gamma, beta, cosh_2t, sinh_2t  );
    double gamma2 = gamma*gamma;
    double omegar2 = omega*omega - gamma2/4;
    double beta_2 = beta*beta;
    double *C = (double*)calloc(2 * 2, sizeof(double));
    //printf("%f %f %f %f\n", C[0],C[1], C[2], C[3] );
    double omega2 = omega *omega;
    double omegar = sqrt(-omegar2);
    C[0] = 1.0/(omega2) + (-4*omega2+gamma2*cosh_2t+2*gamma*omegar*sinh_2t)/(4*omega2* omegar2);
    C[1] =  beta_2/(4*omegar2)*(1.0-cosh_2t);
    C[2] = C[2];
    C[3] = 1.+ (-4*omega2+gamma2*cosh_2t-2*gamma*omegar*sinh_2t)/(4*omegar2);
    //printf("%f %f %f %f\n", C[0],C[1], C[2], C[3]  );
    return C;
}

double* cholesky2(double* A)
{
    double *L = (double*)calloc(2*2, sizeof(double));
    L[0] = sqrt(A[0]+1e-6);
    L[1] = A[1]/L[0];
    L[2] = 0.0;
    L[3] = sqrt(A[3] + 1e-6 - L[1]*L[1]);
    return L;
}

void c_sample_sshm(double U, double mom_U, double A, double Lambda,  double negPrec_prior, double muPrec_prior, double dt, double randn1, double randn2, double* qq, double* pp)
{
    double omega2 = Lambda-negPrec_prior;
    //printf("%f %f %f\n", Lambda, negPrec_prior, omega2 );
    double omega = sqrt(omega2);
    double gamma = A;
    double beta = sqrt(2*A);
    double omegar2 = omega2-gamma*gamma/4;
    double e_gammat2 = exp(-gamma*dt/2);
    double mu = muPrec_prior / omega2;

    double omegar = 0.0;
    double cos_t = 0.0;
    double sin_t = 0.0;
    //printf("inside c_sample_sshm\n");
     if (omegar2 >= 0) {
         omegar = sqrt(omegar2);
         cos_t = cos(omegar*dt);
         sin_t = sqrt(1-cos_t*cos_t);
         double *C = get_SSHM_covariance_pos(omega,gamma, beta, cos_t*cos_t-sin_t*sin_t,2*cos_t*sin_t,e_gammat2*e_gammat2);
         double *L  = cholesky2(C);
         // printf("%f %f %f %f\n",L[0], L[1], L[2], L[3]  );
         // generate random numbers
         double q = randn1*L[0] + randn2 * L[2];
         double p = randn1*L[1] + randn2 * L[3];
         free(C);
         free(L);
         qq[0] = mu+e_gammat2*((U-mu)*cos_t+(mom_U+gamma*(U-mu)/2)*sin_t/omegar)+q;
         pp[0] = e_gammat2*(mom_U*cos_t-((U-mu)*omega2+gamma*mom_U/2)*sin_t/omegar) + p;
     }
     else {
         omegar = sqrt(-omegar2);
         double e_plus= exp(-gamma*dt/2+omegar*dt);
         //double d_plus = -gamma*dt/2+omega*dt;
         //double e_plus = 1. + d_plus + d_plus*d_plus/2;
         double e_minus= exp(-gamma*dt/2-omegar*dt);
         //double d_minus = -gamma*dt/2-omega*dt;
         //double e_minus = 1. + d_minus + d_minus*d_minus/2;
         double cosh_t = 0.5*(e_plus+e_minus);
         double sinh_t = 0.5*(e_plus-e_minus);
         double cosh_2t = 0.5*(e_plus*e_plus+e_minus*e_minus);
         double sinh_2t = 0.5*(e_plus*e_plus-e_minus*e_minus);
         double *C = get_SSHM_covariance_neg(omega, gamma, beta, cosh_2t ,sinh_2t);
         //printf("%f %f %f %f\n", C[0],C[1], C[2], C[3] );
         double *L = cholesky2(C);
         //printf("%f %f %f %f\n",L[0], L[1], L[2], L[3]  );
         // generate random numbers
         double q = randn1*L[0] + randn2 * L[2];
         double p = randn1*L[1] + randn2 * L[3];
         free(C);
         free(L);
         qq[0] = mu+((U-mu)*cosh_t+(mom_U+gamma*(U-mu)/2)*sinh_t/omegar)+q;
         pp[0] = (mom_U*cosh_t-((U-mu)*omega2+gamma*mom_U/2)*sinh_t/omegar) + p;
     }
}


void c_sshm_update(int* ux, double* P, double* mom_P, int* Delta, double* prec_P, double* prior_muprec, double* prior_negprec, double* rands, double A, double dt, int len_ux, int dim)
{
    int idx_u;
    int rnd_idx = 0;
    for (int i = 0; i < len_ux; i++) {
        if (Delta[i] != 0) {
            idx_u = dim * (ux[i] - 1);
            for (int k = 0; k < dim; k++) {
                double q[1];
                double p[1];
                c_sample_sshm(P[idx_u+k],mom_P[idx_u+k],A,prec_P[k],prior_negprec[idx_u+k],prior_muprec[idx_u+k],Delta[i]*dt,rands[rnd_idx+1],rands[rnd_idx+2],q,p);
                rnd_idx = rnd_idx + 2;
                P[idx_u+k] = q[0];
                mom_P[idx_u+k] = p[0];
            }
        }
    }
}


void c_update_para_sghmc_sshm(int* ux, double* P, double* mom_P, double* g_sum_P, double* prec_P, double* prior_muprec, double* prior_negprec, double* rands, int* Delta,
                   double ka, double stepsz, double sqrtstepsz, double A, int len_ux, int dim)
{
    int idx_u;
    int rnd_idx = 0;
    double prec;
    //printf("Inside c function\n" );
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        for (int k = 0; k < dim; k++) {
            if (Delta[i] != 0) {
                double q[1];
                double p[1];
                c_sample_sshm(P[idx_u+k],mom_P[idx_u+k],A,prec_P[k],prior_negprec[idx_u+k],prior_muprec[idx_u+k],Delta[i]*stepsz,rands[rnd_idx+1],rands[rnd_idx+2],q,p);
                rnd_idx = rnd_idx + 2;
                P[idx_u+k] = q[0];
                mom_P[idx_u+k] = p[0];
            }

            prec =   prec_P[k] - prior_negprec[idx_u + k];
            mom_P[idx_u + k] += stepsz * (ka * g_sum_P[idx_u + k] - A * mom_P[idx_u +k] - ( prec*(P[idx_u + k] - prior_muprec[idx_u + k]/prec)  )) + sqrtstepsz * rands[rnd_idx++];
            P[idx_u + k] += stepsz * mom_P[idx_u + k ];
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}




void c_update_para_sgd_prior(int* ux, double* P, double* g_sum_P, double* prec_P, double* prior_muprec, double* prior_negprec, double* inv_pr_pick, double ka, double halfstepsz, int len_ux, int dim)
{
    int idx_u;
    double inv_pr_pick_i;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        inv_pr_pick_i = inv_pr_pick[ux[i]-1];
        for (int k = 0; k < dim; k++) {
            P[idx_u + k] += halfstepsz * (ka * g_sum_P[idx_u + k] - inv_pr_pick_i * (( prec_P[k] - prior_negprec[idx_u + k]) *  P[idx_u + k] - prior_muprec[idx_u + k])  );
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}


void c_add_prior_grad(double*grad, double* P, double* prec_P, double* prior_muprec, double* prior_negprec, int dim, int len_P, double blocksize, double beta)
{
    int idx_u;
    for (int i=0; i < len_P; i++){
        idx_u = dim * i;
        for (int k=0; k < dim; k++){
            grad[idx_u+k] -= (prec_P[k]/(blocksize*beta) - prior_negprec[idx_u + k]) *  P[idx_u + k] - prior_muprec[idx_u + k];
        }
    }
}


void c_update_para_m1(int* ux, double* P, double* g_sum_P, double* mu_p, double* La_p, double* inv_pr_pick, double* rands, double ka, double halfstepsz, double sqrtstepsz, int len_ux, int dim, int batch_sz)
{
    int idx_u;
    int rnd_idx = 0;
    double grad_prior_ik;
    double inv_pr_pick_i;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        inv_pr_pick_i = inv_pr_pick[ux[i]-1];
        for (int k = 0; k < dim; k++) {
            grad_prior_ik = 0.0;
            for (int j = 0; j < dim; j++) {
                grad_prior_ik += La_p[j*dim + k] * (P[idx_u + j] - mu_p[j]);
            }
            P[idx_u + k] += (halfstepsz * (ka * g_sum_P[idx_u + k] - inv_pr_pick_i * grad_prior_ik) + sqrtstepsz * rands[rnd_idx++]) ;
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}


void c_update_para_sgd(int* ux, double* P, double* g_sum_P, double* prec_P, double* inv_pr_pick,
                       double ka, double halfstepsz, int len_ux, int dim)
{
    int idx_u;
    double inv_pr_pick_i;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        inv_pr_pick_i = inv_pr_pick[ux[i]-1];
        for (int k = 0; k < dim; k++) {
            P[idx_u + k] += halfstepsz * (ka * g_sum_P[idx_u + k] - prec_P[k] * inv_pr_pick_i * P[idx_u + k]);
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}

void c_update_para_sgd_m1(int* ux, double* P, double* g_sum_P, double* mu_p, double* La_p, double* inv_pr_pick, double ka, double halfstepsz, int len_ux, int dim, int batch_sz)
{
    int idx_u;
    /*double grad_prior_ik;*/
    double inv_pr_pick_i;
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        inv_pr_pick_i = inv_pr_pick[ux[i]-1];
        for (int k = 0; k < dim; k++) {
            /*
             *grad_prior_ik = 0;
             *for (int j = 0; j < dim; j++) {
             *    grad_prior_ik += La_p[j*dim + k] * (P[idx_u + j] - mu_p[j]);
             *}
             */
            /*P[idx_u + k] += halfstepsz * (ka * g_sum_P[idx_u + k] - inv_pr_pick_i * grad_prior_ik) ;  */
            P[idx_u + k] += halfstepsz * (ka * g_sum_P[idx_u + k] - inv_pr_pick_i * La_p[k*dim + k] * (P[idx_u + k] - mu_p[k])) ;
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}

/*
 *void c_update_para_sgd_m1_(int* ux, double* P, double* g_sum_P, double* grad_prior_P, double* inv_pr_pick, double ka, double halfstepsz, int len_ux, int dim, int batch_sz)
 *{
 *    int idx_u, idx_uk;
 *    [>double grad_prior_ik;<]
 *    double inv_pr_pick_i;
 *    for (int i = 0; i < len_ux; i++) {
 *        idx_u = dim * (ux[i] - 1);
 *        inv_pr_pick_i = inv_pr_pick[ux[i]-1];
 *        for (int k = 0; k < dim; k++) {
 *            [>grad_prior_ik = 0;<]
 *            for (int j = 0; j < dim; j++) {
 *                grad_prior_ik += La_p[j*dim + k] * (P[idx_u + j] - mu_p[j]);
 *            }
 *            idx_uk = idx_u + k;
 *            P[idx_uk] += halfstepsz * (ka * g_sum_P[idx_uk] - inv_pr_pick_i * grad_prior_P[idx_uk]);
 *            g_sum_P[idx_uk] = 0.0;  // initialize!
 *        }
 *    }
 *}
 */


void c_update_para_sgd_mom_m1(int* ux, double* P, double* g_mom_P, double* g_sum_P, double* mu_p, double* La_p, double ka, double halfstepsz, int len_ux, int dim, int batch_sz, float mom_rate)
{
    int idx_u;
    double grad_prior_ik;
    /*double inv_pr_pick_i;*/
    for (int i = 0; i < len_ux; i++) {
        idx_u = dim * (ux[i] - 1);
        /*inv_pr_pick_i = inv_pr_pick[ux[i]-1];*/
        for (int k = 0; k < dim; k++) {
            grad_prior_ik = 0;
            for (int j = 0; j < dim; j++) {
                grad_prior_ik += La_p[j*dim + k] * (P[idx_u + j] - mu_p[j]);
            }
            // update gradient momentum
            g_mom_P[idx_u + k] = mom_rate * g_mom_P[idx_u + k] + halfstepsz * (ka * g_sum_P[idx_u + k] - grad_prior_ik);
            // update parameter
            P[idx_u + k] += g_mom_P[idx_u + k];
            g_sum_P[idx_u + k] = 0.0;  // initialize!
        }
    }
}


void c_rmse_avg_m2(double* testset, double* P, double* Q, double* A, double* B, double* avg_pred, int iter, double min_rate, double max_rate, double mean_rate, int is_burnin, int sz_testset, int dim, double* avg_rmse, double* cur_rmse, int* avg_cnt)
{
    double r_i, p_i, tmp;
    int u_i, i_i;

    if (!is_burnin) avg_cnt[0] += 1;

    for (int i = 0; i < sz_testset; i++) {
        u_i = testset[i] - 1;
        i_i = testset[sz_testset + i] - 1;
        r_i = testset[2*sz_testset + i];
        p_i = dot_product(&P[dim * u_i], &Q[dim * i_i], dim) + A[u_i] + B[i_i] + mean_rate;
        if (p_i < min_rate) p_i = min_rate;
        if (p_i > max_rate) p_i = max_rate;
        if (!is_burnin) {
            avg_pred[i] = (1 - 1./avg_cnt[0]) * avg_pred[i] + 1./avg_cnt[0] * p_i;
            tmp = r_i - avg_pred[i];
            avg_rmse[0] += tmp * tmp;
        }
        else {
            avg_pred[i] = p_i;
        }
        tmp = r_i - p_i;
        cur_rmse[0] += tmp * tmp;
    }
    avg_rmse[0] = sqrt(avg_rmse[0] / sz_testset);
    cur_rmse[0] = sqrt(cur_rmse[0] / sz_testset);
}


void c_predict(double* testset, double* P, double* Q, double* A, double* B, double* pred, double min_rate, double max_rate, double mean_rate, int sz_testset, int dim)
{
    double p_i;
    int u_i, i_i;
    for (int i = 0; i < sz_testset; i++) {
        u_i = testset[i] - 1;
        i_i = testset[sz_testset + i] - 1;
        p_i = dot_product(&P[dim * u_i], &Q[dim * i_i], dim) + A[u_i] + B[i_i] + mean_rate;
        if (p_i < min_rate) p_i = min_rate;
        if (p_i > max_rate) p_i = max_rate;
        pred[i] = p_i;
    }
}



void c_rmse_avg_m1(double* testset, double* P, double* Q, double* avg_pred, int iter, double min_rate, double max_rate, double mean_rate, int is_burnin, int sz_testset, int dim, double* avg_rmse, double* cur_rmse, int* avg_cnt)
{
    double r_i, p_i, tmp;
    int u_i, i_i;

    if (!is_burnin) avg_cnt[0] += 1;

    for (int i = 0; i < sz_testset; i++) {
        u_i = testset[i] - 1;
        i_i = testset[sz_testset + i] - 1;
        r_i = testset[2*sz_testset + i];
        p_i = dot_product(&P[dim * u_i], &Q[dim * i_i], dim) + mean_rate;
        if (p_i < min_rate) p_i = min_rate;
        if (p_i > max_rate) p_i = max_rate;
        if (!is_burnin) {
            avg_pred[i] = (1 - 1./avg_cnt[0]) * avg_pred[i] + 1./avg_cnt[0] * p_i;
            tmp = r_i - avg_pred[i];
            avg_rmse[0] += tmp * tmp;
        }
        else {
            avg_pred[i] = p_i;
        }
        tmp = r_i - p_i;
        cur_rmse[0] += tmp * tmp;
    }
    avg_rmse[0] = sqrt(avg_rmse[0] / sz_testset);
    cur_rmse[0] = sqrt(cur_rmse[0] / sz_testset);
}
