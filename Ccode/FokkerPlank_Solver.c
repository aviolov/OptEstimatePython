#include "FokkerPlank_Solver.h"


void _fortetError(const double * ts, const int num_steps, double * Is, const int num_intervals,
				const double * abgthphi,
				double *error) {
	const double alpha = abgthphi[0];
	const double beta = abgthphi[1];
	const double gamma = abgthphi[2];
	const double theta = abgthphi[3];
	const double phi = abgthphi[4];
//	printf("Fortet: theta=%.2f,gamma=%.2f\n", theta, gamma);

	const double TAU_TOLERANCE = 1e-5;
	const double psi = atan(theta);
	//Sort is:
	gsl_sort(Is, 1, num_intervals);

	double lt, lI, ltau, vthforward, vthbackward;
	double numerator, denominator, lhs;
	double max_lhs = .0;
	double normalizing_const = 1.0*num_intervals;

	double exp_factor;
	for (int tidx = 0; tidx < num_steps; ++tidx) {
		lt = ts[tidx];
		vthforward = 1. - (alpha*(1 - exp(-lt)) +
						gamma / sqrt(1.+theta*theta) * ( sin ( theta *  ( lt + phi) - psi)
												-exp(-lt)*sin(phi*theta - psi) ));

		//Calculate LHS:
		numerator = vthforward * sqrt(2.);
		denominator = beta * sqrt(1. - exp(-2.*lt));

		lhs = gsl_cdf_gaussian_Q(numerator/denominator, 1.0);

		max_lhs = GSL_MAX(lhs, max_lhs);

		error[tidx] = lhs;

		//Calculate RHS:
		for (int idx = 0; idx < num_intervals; ++idx){
			lI = Is[idx];
			if(lI >= (lt - TAU_TOLERANCE)) break;

			ltau = lt - lI;

			exp_factor = exp(-lI);
			vthbackward = 1. - (alpha*(1. - exp_factor) +
					gamma / sqrt(1.+theta*theta) * ( sin ( theta *  ( lI + phi) - psi)
											-exp_factor*sin(phi*theta - psi) ));
			exp_factor = exp(-ltau);
			numerator = (vthforward - vthbackward*exp_factor)*sqrt(2.);
			denominator = beta * sqrt(1. - exp_factor*exp_factor);

			error[tidx] -= gsl_cdf_gaussian_Q(numerator/denominator, 1.0) / normalizing_const;

		}//end rhs (intervals) loop
	}//end error (time steps) loop

	//scale by max_lhs:
	for (int tidx = 0; tidx < num_steps; ++tidx) {
		error[tidx] = fabs(error[tidx]) / max_lhs;
	}
}

void _fortetRHS(const double * ts, const int num_steps, double * Is, const int num_intervals,
				const double * abgthphi,
				double *rhs) {
	const double alpha = abgthphi[0];
	const double beta = abgthphi[1];
	const double gamma = abgthphi[2];
	const double theta = abgthphi[3];
	const double phi = abgthphi[4];

	const double TAU_TOLERANCE = 1e-5;
	const double psi = atan(theta);
//	sort is
	gsl_sort(Is, 1, num_intervals);

	double lt, lI, ltau, vthforward, vthbackward,numerator, denominator;

	for (int tidx = 0; tidx < num_steps; ++tidx) {
		lt = ts[tidx];
		rhs[tidx] = .0;

		vthforward = 1. - (alpha*(1 - exp(-lt)) +
						gamma / sqrt(1.+theta*theta) * ( sin ( theta *  ( lt + phi) - psi)
												-exp(-lt)*sin(phi*theta - psi) ));

		for (int idx = 0; idx < num_intervals; ++idx){
			lI = Is[idx];
			if(lI >= (lt - TAU_TOLERANCE)) break;

			ltau = lt - lI;

			vthbackward = 1. - (alpha*(1 - exp(-lI)) +
					gamma / sqrt(1.+theta*theta) * ( sin ( theta *  ( lI + phi) - psi)
											-exp(-lI)*sin(phi*theta - psi) ));

			numerator = (vthforward - vthbackward*exp(-ltau))*sqrt(2.);
			denominator = beta * sqrt(1. - exp(-2*ltau));

			rhs[tidx] += gsl_cdf_gaussian_Q(numerator/denominator, 1.0);

		}//end per time loop
	}//end all times loop
}

void _simulateSDE(const double * abgth, const int N_spikes, const double dt,
					double * Ss) {
	const double V_THRESH = 1.0;

	double alpha, beta, gamma, theta;
	alpha = abgth[0]; beta= abgth[1]; gamma =abgth[2]; theta = abgth[3];

	int recorded_spikes = 0;

	double v = .0;
	double sqrt_dt = sqrt(dt);

	//RNG stuff:
    const gsl_rng_type * T;
    gsl_rng * G;
    /* create a generator chosen by the environment variable GSL_RNG_TYPE */
    gsl_rng_env_setup();
    T = gsl_rng_default;
    G = gsl_rng_alloc (T);

    struct timeval tv;
    gettimeofday(&tv,NULL);
    srand((tv.tv_sec * 1000) + (tv.tv_usec / 1000));
    unsigned long int S = rand();
//    printf("S=%d\n",S);
    gsl_rng_set (G, S);

    double dB, dv;
    double t = .0;
	while (recorded_spikes < N_spikes){
				dB =  gsl_ran_gaussian_ziggurat(G, sqrt_dt);
	            dv = ( alpha - v + gamma*sin(theta*t) )*dt  + beta*dB;

	            v += dv;
	            t += dt;
	            if (v >= V_THRESH){
	                Ss [recorded_spikes++] = t;
	                v = .0;
	            }
	}

	gsl_rng_free(G);
}
//1870241891

void _solveFP(const double * abgth,
				const double * phis, const int N_phi,
				const double * ts, const int num_steps, //num_steps is not the best way to call this variable it is actually num_epochs or num_tks
				const double * xs, const int num_nodes,
				double *** Fs) {

//	printf("a,b,g,t = %.2f %.2f %.2f %.2f", abgth[0],abgth[1],abgth[2],abgth[3]);
//	double *** Fs;
//	= new double[N_phi][num_steps][num_nodes];
//	double * contiguousAllocationPtr = malloc(N_phi*num_steps*num_nodes * sizeof(double));
//	Fs = malloc(N_phi * sizeof(double**));

//	for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx) {
//		for (int t_idx = 0; t_idx < num_steps; ++t_idx) {
//			for (int x_idx = 0; x_idx < num_nodes; ++x_idx) {
//				Fs[phi_idx][t_idx][x_idx] = phis[phi_idx]*ts[t_idx]*xs[x_idx];
//			}
//		}
//	}
	//RIP PARAMS:
	double alpha, beta, gamma, theta;
	double dx, dt;
	alpha = abgth[0]; beta= abgth[1]; gamma =abgth[2]; theta = abgth[3];
	dt = ts[1]-ts[0]; //or pass explicitly??? (nein!)
	dx = xs[1]-xs[0]; //or pass explicitly??? (nein!)

	printf("FP: alpha=%.2f, beta=%.2f,gamma=%.2f,theta=%.2f\n",
			alpha, beta, gamma, theta);

	double D =  beta * beta / 2.; //the diffusion coeff
	double dx_sqrd = dx * dx;

	//Set BCs, ICs:
	setICs(Fs, phis, N_phi, xs, num_nodes);
	setBCs(Fs, phis, N_phi, ts, num_steps);

//  e = ones(self._num_nodes() - 1);
//	d_on = D * dt / dx_sqrd;
//	centre_diag = e + d_on;
//	centre_diag[-1] = 1.0;
//	M.setdiag(centre_diag)

	double t_prev, t_next;
	double * U_prev; double * U_next;	double * F_prev; double * F_next;
	double * L_prev; double * RHS;
	int num_active_nodes = num_nodes -1; //Recall that the first node (F[0] is identically zero!!!) only from x_k = 2...N_x are they active
	//TODO: create an array loop for the following:
	U_prev = (double *)malloc(num_active_nodes*sizeof(double));
	U_next = (double *)malloc(num_active_nodes*sizeof(double));
	F_prev = (double *)malloc(num_nodes*sizeof(double)); //!!!
	F_next = (double *)malloc(num_active_nodes*sizeof(double));
	L_prev = (double *)malloc(num_active_nodes*sizeof(double));
	RHS    = (double *)malloc(num_active_nodes*sizeof(double));

    //AllOCATE MASS MTX (diagonals):
	double (* M_l), (*M_c), (*M_u); //lower, central and upper diagonal respectively
	M_l = (double *)malloc(num_active_nodes*sizeof(double));
	M_c = (double *)malloc(num_active_nodes*sizeof(double));
	M_u = (double *)malloc(num_active_nodes*sizeof(double));

	double max_F_next = .0;
	double phi, x_cur; //Uninitialized!
	for (int tk = 1; tk < num_steps; ++tk) {
           t_prev = ts[tk-1];
           t_next = ts[tk];
           for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx) {
        	   phi = phis[phi_idx];

        	   //reload the variables for this time step:
        	   //Rip the previous time solution:
        	   for (int x_idx = 0; x_idx < num_nodes; ++x_idx)
        		   F_prev[x_idx] = Fs[phi_idx][tk-1][x_idx];

			   //Form the advection coefficient:
			   for (int x_idx = 0; x_idx < num_active_nodes; ++x_idx) {
        		   x_cur = xs[x_idx+1];
				   U_prev[x_idx] = -(alpha - x_cur + gamma * sin(theta * (t_prev + phi) ));
				   U_next[x_idx] = -(alpha - x_cur + gamma * sin(theta * (t_next + phi) ));
        	   }

			   //Form operator term at the previous step:
        	   for (int x_idx = 0; x_idx < num_active_nodes-1; ++x_idx) {
        		   L_prev[x_idx] =  U_prev[x_idx] * (F_prev[x_idx+2] - F_prev[x_idx]) / 2. / dx 			//advection
                                  + D * (F_prev[x_idx+2] -  2*F_prev[x_idx+1] + F_prev[x_idx]) / dx_sqrd;   //diffusion

        	   }
        	   //Specifically handle right side conditions:
        	   L_prev[num_active_nodes-1] =   U_prev[num_active_nodes-1]*(F_prev[num_nodes-1] - F_prev[num_nodes-2]) / dx
        			   	   	   	   	   	  + D * (-F_prev[num_nodes-1] + F_prev[num_nodes-2]) / dx_sqrd; //the form of the diffusion term comes from extrapolating the zero Neumann BCs:

        	   //Finally form the RHS:
        	   for (int x_idx = 0; x_idx < num_active_nodes-1; ++x_idx)
        		   RHS[x_idx]= F_prev[x_idx+1] + .5 * dt * L_prev[x_idx];

        	   //and impose the right BCs:
               RHS[num_active_nodes-1] = 0.;

               //Set the MASS MTX:
               //Main diagonal: Carefully read the Thomas Algo index specs!!!
               double d_on  = -D * dt / dx_sqrd;
               double d_off = .5 * D * dt / dx_sqrd;
               double u_factor = .5 * dt / dx / 2.;

               for (int x_idx = 0; x_idx < num_active_nodes-1; ++x_idx) {
            	   //Central diagonal
               		M_c[x_idx] = 1.0 - d_on;
               	   //Lower diagonal: Carefully read the THomas Algo index specs (it only uses the values from M_l[1] onwards !!!
               	   M_l[x_idx+1] = -(d_off - u_factor * U_next[x_idx+1]);
               	   //Upper diagonal: Carefully read the THomas Algo index specs!!!
				   M_u[x_idx]   = -(d_off + u_factor * U_next[x_idx]);
               }
//			  Set the Neumann BCs:
			  M_l[num_active_nodes-1] = -1.0; //
			  M_c[num_active_nodes-1] = 1.0;


               //Thomas Solve it:
               thomasSolve(num_active_nodes, M_l, M_c, M_u, RHS, F_next);

               //Store solution:
               for (int x_idx = 1; x_idx < num_nodes; ++x_idx) {
            	   double lF = F_next[x_idx-1];
            	   max_F_next = fmax(max_F_next, lF);
            	   Fs[phi_idx][tk][x_idx] = lF;
               }
		}//end per-phi loop
//      TODO: Break out of time-loop?
//		if max_F_next< 1e-4:
//			break // from time-loop (remember to set the rest of F to zero!!!
	}//end time loop
}//_solveFP(...)


void setICs(double *** Fs, const double * phis, const int N_phi, const double * xs, const int num_nodes) {
	for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx) {
			for (int x_idx = 0; x_idx < num_nodes; ++x_idx) {
				double x = xs[x_idx];
				(x>= 0)?
					(Fs[phi_idx][0][x_idx] = 1.0) :
						(Fs[phi_idx][0][x_idx] = .0) ;
			}
	}
}

void setBCs(double *** Fs, const double * phis, const int N_phi, const double * ts, const int num_steps) {
	for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx) {
			for (int t_idx = 0; t_idx < num_steps; ++t_idx) {
					Fs[phi_idx][t_idx][0] = .0 ;
			}
	}
}



void _solve_f(const double mutausigma[3 ],
			  const double * alphas,
		      const double * ts, const int num_steps,
		      const double * xs, const int num_nodes,
		      double ** fs) {
	/* we use a C-N method to forward solve a basic
		Fokker-Planck LIF-based hitting time problem:
	 *
	 * 			Df_xx - (Uf)_x = f_t
	 * */

	//RIP PARAMS:
	double mu, tauchar, sigma;
	double dt, dx, dx_squared;
	mu = mutausigma[0]; tauchar = mutausigma[1]; sigma=mutausigma[2];
	dt = ts[1] - ts[0];
	dx = xs[1] - xs[0];
	dx_squared = dx*dx;

	// Linear System Variables:
	size_t num_active_nodes = num_nodes - 1;
//	size_t num_inner_nodes = num_nodes - 2;

	double D = sigma*sigma/2.0;
	double u_l, u_r, d_on, d_off;
	d_on  = dt * D / dx_squared;
	d_off = dt * D / dx_squared;
	//u is allocated on the fly in the time / space loop
	 //AllOCATE MASS MTX (diagonals):
	double * M_l, *M_c, *M_u, *RHS;//lower, central and upper diagonal respectively and the RHS vector
	M_l = (double *)malloc(num_active_nodes*sizeof(double));
	M_c = (double *)malloc(num_active_nodes*sizeof(double));
	M_u = (double *)malloc(num_active_nodes*sizeof(double));
	RHS = (double *)malloc(num_active_nodes*sizeof(double));

	//Declare (and allocate) loop variables:
	double * f_prev, * f_current;
	double alpha_prev, alpha_current;
	double * U_prev, *U_current;
	f_prev =  malloc(num_nodes*sizeof(double));
	f_current = malloc(num_active_nodes*sizeof(double));
	U_prev = malloc(num_nodes*sizeof(double));
	U_current = malloc(num_nodes*sizeof(double));

	//set (Upper) BCs (homogeneous):
	for (int tk = 0; tk < num_steps; ++tk) {
		fs[tk][num_active_nodes] = 0.0;
	}
	//set ICs:
	const double IC_WIDTH = 0.01;
	double norm_const = 0.0;
	for (int xdx = 0; xdx < num_active_nodes; ++xdx) {
		fs[0][xdx] =  exp(-xs[xdx]*xs[xdx] /  (2.*IC_WIDTH*IC_WIDTH) );
		norm_const+= fs[0][xdx];
	}
	norm_const*=dx;
	for (int xdx = 0; xdx < num_nodes; ++xdx) {
		fs[0][xdx] =  fs[0][xdx] / norm_const;
	}

	// lower BCs:
	RHS[0] = 0.0;

	int tk_prev;
	double sde_field;
	//MAIN LOOP:
	for (int tk = 1; tk < num_steps; ++tk) {
		tk_prev = tk-1;
		alpha_prev    = alphas[tk_prev];
		alpha_current = alphas[tk];
		//rip prev soln:
		for (int xdx = 0; xdx < num_nodes; ++xdx) {
			//TODO: just reassign ptr:
			f_prev[xdx] = fs[tk_prev][xdx];
		}
		//Calculate U:
		for (int xidx = 0; xidx < num_nodes; ++xidx) {
			// U = ( alpha_prev + (mu - xs / tau_char) )
			sde_field  = (mu - xs[xidx])/tauchar;
			U_prev[xidx]    = alpha_prev    + sde_field;
			U_current[xidx] = alpha_current + sde_field;
		}

		//Form RHS:
		for (int xidx = 1; xidx < num_active_nodes; ++xidx) {
			RHS[xidx] = f_prev[xidx] +
						dt*0.5* ( D * ( f_prev[xidx+1] - 2.*f_prev[xidx] + f_prev[xidx-1]) / dx_squared -
		 					     (U_prev[xidx+1]*f_prev[xidx+1] - U_prev[xidx-1]*f_prev[xidx-1] ) / (2.0 * dx) );
		}

		//Form M: i.e  M_l, M_c, M_u
		//set inner matrix entries: xidx starts at 1:
		for (int xdx = 1; xdx < num_active_nodes; ++xdx) {
			u_l = dt * U_current[xdx-1] / (2.*dx);
			u_r = dt * U_current[xdx+1] / (2.*dx);
			//Lower diagonal: Carefully read the Thomas Algo specs (it only uses the values from M_l[1] onwards !!!
			M_l[xdx] = -.5*(d_off + u_l);
		    //Central diagonal
			M_c[xdx] = 1. + d_on;
		    //Upper diagonal: Carefully read the THomas Algo index specs!!!
		    M_u[xdx] = -.5*(d_off - u_r);
//		    printf("%.4f,%.4f,%.4f : %.4f\n", M_l[xdx], M_c[xdx], M_u[xdx], RHS[xdx]);
		}
		//Lower BCs (homogeneous Neumann:
		//D f' - Uf = 0
		M_c[0] = -U_current[0] - D / dx;  M_u[0] =  D / dx;


        //Thomas Solve it:
        thomasSolve(num_active_nodes,
        		    M_l, M_c, M_u,
        		    RHS, f_current);

		//populate the solution field:
		for (int xidx = 0; xidx < num_active_nodes; ++xidx) {
			// TODO pass fs[tk] directly to the Thomas Solver//
			//i.e. pass fs[tk][0] or &fs[tk][0] or vs[tk*num_nodes]
			fs[tk][xidx] = f_current[xidx];
		}

	} //END TIME LOOP

	// clean up:
	free(M_l); free(M_c); free(M_u);
	free(RHS);
	free(f_prev); free(f_current); free(U_prev); free(U_current);
}

void _solve_p(){

}


void thomasSolve(int n, double *M_l, double *M_c, double *M_u, double *v, double *x)
{        /**
         * n - number of equations
         * M_l - sub-diagonal (means it is the diagonal below the main diagonal) -- indexed from 1..n-1
         * M_c - the main diagonal
         * M_u - sup-diagonal (means it is the diagonal above the main diagonal) -- indexed from 0..n-2
         * v   - right-hand side
         * x   - the answer (return val)
         */
		//Elimination:
        for (int idx = 1; idx < n; ++idx){
                double m = M_l[idx]/M_c[idx-1];
                M_c[idx] = M_c[idx] - m * M_u[idx - 1];

                v[idx] = v[idx] - m*v[idx-1];
        }

        x[n-1] = v[n-1]/M_c[n-1];

        for (int i = n-2; i >= 0; --i)
                x[i] = (v[i] - M_u[i] * x[i+1]) / M_c[i];
}

