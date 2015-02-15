/*
 * FokkerPlank_Solver.h
 *
 *  Created on: Oct 2, 2012
 *      Author: alex
 */

#ifndef FOKKERPLANK_SOLVER_H_
#define FOKKERPLANK_SOLVER_H_

#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sort_double.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_cdf.h>

//Linux / gcc specific headers:
//#include <time.h>
#include <sys/time.h>

//API Functions
void _solveFP(const double * abgth,
				const double * phis, const int N_phi,
				const double * ts, const int num_steps,
				const double * xs, const int num_nodes,
				double ***Fs);

void _simulateSDE(const double * abgth, const int N_spikes, const double dt,
					double * Ss);

void _fortetRHS(const double * ts, const int num_steps, double * Is, const int num_intervals,
				const double * abgthphi,
				double *rhs);
void _fortetError(const double * ts, const int num_steps, double * Is, const int num_intervals,
				const double * abgthphi,
				double *error);

void _solve_f(const double mutausigma[3 ],
		const double * alphas,
		      const double * ts, const int num_steps,
		      const double * xs, const int num_nodes,
		      double ** fs);

//HELPER FUNCTIONS:
void setICs(double *** Fs, const double * phis, const int N_phi, const double * xs, const int num_nodes);
void setBCs(double *** Fs, const double * phis, const int N_phi, const double * ts, const int num_steps);
void thomasSolve(int n, double *M_l, double *M_c, double *M_u, double *v, double *x);

#endif /* FOKKERPLANK_SOLVER_H_ */
