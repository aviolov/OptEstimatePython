/*
 * Driver.c
 *
 *  Created on: Oct 3, 2012
 *      Author: alex
 */


/* to valgrind this program do:
 *   valgrind --leak-check=yes ./Driver (no args are needed, but if o/w put  arg1 arg2 here
 */

#include <stdio.h>
#include <stdlib.h>

#include "FokkerPlank_Solver.h"

#define N_phi  2

void solveFDriver() {
	const int num_steps  = 3;
	const int num_nodes  = 4;

	double *** Fs;

	//  HOW-TO Allocate a Contiguous 3D Array:
	double*contiguousAllocationPtr = malloc(N_phi * num_steps * num_nodes * sizeof(double));
	Fs = malloc(N_phi * sizeof(double**));
	for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx) {
		Fs[phi_idx] = malloc(num_steps* sizeof(double*) );
		for (int t_idx = 0; t_idx < num_steps; ++t_idx)
			Fs[phi_idx][t_idx] = contiguousAllocationPtr + (phi_idx * num_steps*num_nodes ) + (t_idx*num_nodes);
	}

	double abgth [4] = {1.0,1.0,1.0,2.0};

	double phis[N_phi]= {.05, 3.14/2.0};
	double ts[3]=  {.0, .05, .1};
	double xs[4]= {-.25, .25, .75, 1.0};

	_solveFP(abgth, phis, N_phi, ts, num_steps, xs , num_nodes, Fs);

	for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx) {
		printf("phi = %f: \n", phis[phi_idx]);
		for (int t_idx = 0; t_idx < num_steps; ++t_idx) {
			for (int x_idx = 0; x_idx < num_nodes; ++x_idx) {
				printf("%f  ", Fs[phi_idx][t_idx][x_idx]);
			}
			printf("\n");
		}
	}
}


void allocate3DExample() {
	//  Array 3 Dimensions
	int x = 4, y = 5, z = 6;

	//  Array Iterators
	int i, j, k;

	//  Allocate 3D Array
	int *allElements = malloc(x * y * z * sizeof(int));
	int ***array3D = malloc(x * sizeof(int **));

	for(i = 0; i < x; i++)
	{
		array3D[i] = malloc(y * sizeof(int *));

		for(j = 0; j < y; j++)
		{
			array3D[i][j] = allElements + (i * y * z) + (j * z);
		}
	}

	//  Access array elements
	for(i = 0; i < x; i++)
	{
		printf("%d\n", i);

		for(j = 0; j < y; j++)
		{
			printf("\n");

			for(k = 0; k < z; k++)
			{
				array3D[i][j][k] = (i * y * z) + (j * z) + k;
				printf("\t%d", array3D[i][j][k]);
			}
		}

		printf("\n\n");
	}

	//  Deallocate 3D array
	free(allElements);
	for(i = 0; i < x; i++)
	{
		free(array3D[i]);
	}
	free (array3D);
}

void solve_f_Driver() {
	const int num_steps = 2;


	double mutausigma[3] = {0., 2e5, 1.};
	double alphas[2] = {0., 0.};
	double ts[2] = {0, 1};
//	const int num_nodes = 6;
//	double xs[6] = {-.25, .0, .25, .5, .75 ,1};
	const int num_nodes = 4;
	double xs[4] = {-.5, .0,  .5 , 1};

	//Raw data for the return array:
	double (** fs);

	//  Allocate a 2D Array  (contiguously!)!!!
	fs = malloc(num_steps * sizeof(double *));
	fs[0] = malloc(num_steps*num_nodes * sizeof(double));
	for(size_t idx = 1; idx < num_steps; ++idx) {
		fs[idx] = fs[0] + idx * num_nodes;
	}

	_solve_f(mutausigma, alphas,
			ts, num_steps,
			xs, num_nodes,
		    fs);

	for (int tdx = 0; tdx < num_steps; ++tdx) {
		for (int xdx = 0; xdx < num_nodes; ++xdx) {
			printf("%.2f \t", fs[tdx][xdx]);
		}
		printf(" |\n");
	}

	printf("DONE\n");
}


double solve_f_g_Driver(int refine_factor) {
	const int num_nodes = 250*refine_factor;
	const int num_steps = 500*refine_factor;
	double dt = 0.0251889168766/refine_factor;
	double dx = 0.025/refine_factor;


	double mutausigma[3] = {0., 1.11, 1.};

	double * ts = malloc(num_steps*sizeof(double)) ;
	double * xs = malloc(num_nodes*sizeof(double)) ;
	double * alphas = malloc(num_steps*sizeof(double)) ;
	alphas[0] = 1.0;
	ts[0]= .0;
	for (int tdx = 1; tdx < num_steps; ++tdx) {
		ts[tdx] = ts[tdx-1]+dt;
		alphas[tdx] = 1.0;
	}
	xs[num_nodes-1] = 1.0;
	for (int xdx = num_nodes-2; xdx >= 0; --xdx) {
		xs[xdx] = xs[xdx+1] - dx;
	}

	//Raw data for the return array:
	double (** fs);

	//  Allocate a 2D Array  (contiguously!)!!!
	fs = malloc(num_steps * sizeof(double *));
	fs[0] = malloc(num_steps*num_nodes * sizeof(double));
	for(size_t idx = 1; idx < num_steps; ++idx) {
		fs[idx] = fs[0] + idx * num_nodes;
	}

	//MAIN CALL:
	_solve_f(mutausigma, alphas,
			ts, num_steps,
			xs, num_nodes,
		    fs);

	double * gs = malloc(num_steps*sizeof(double)) ;
	double norm_const = .0;
	for (int tdx = 0; tdx < num_steps; ++tdx) {
		gs[tdx] = -(fs[tdx][num_nodes-1] - fs[tdx][num_nodes-2])/dx / 2.0; // sigma = 1 => D = 1/2
		norm_const += gs[tdx]*dt;
	}

	free(gs);
	free(ts);
	free(xs);
	free(alphas);

	free( (void *) fs[0] );
	free( (void *) fs );

	return norm_const;

	printf("DONE\n");
}

void solve_f_g_Harness(){
	double norm_const = 0.;
	for (int rf = 1; rf < 8; ++rf) {
		norm_const = solve_f_g_Driver(rf);
		printf("%d: sum(gs) = %.8f\n",rf, norm_const);
	}
}


int main(int argc, char **argv) {
//	solveFDriver();

//	solve_f_Driver();
	solve_f_g_Harness();

}


