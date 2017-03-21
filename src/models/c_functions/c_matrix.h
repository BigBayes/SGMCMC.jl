#include <math.h>
#include <stdlib.h>

double *cholesky(double *A, int n) {
    double *L = (double*)calloc(n * n, sizeof(double));
    if (L == NULL)
        exit(EXIT_FAILURE);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < (i+1); j++) {
            double s = 0;
            for (int k = 0; k < j; k++)
                s += L[i * n + k] * L[j * n + k];
            L[i * n + j] = (i == j) ?
                        sqrt(A[i * n + i] - s) :
                        (1.0 / L[j * n + j] * (A[i * n + j] - s));
    }
    return L;
}

int sayhi()
{
    printf("hihi3\n");
    return 0;
}

