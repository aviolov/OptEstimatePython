#include <Python.h>            /* Python as seen from C */
#include <numpy/arrayobject.h> /* NumPy  as seen from C */
#include <math.h>
#include <stdio.h>             /* for debug output */
#include <NumPy_macros.h>
// local code:
#include "src/FokkerPlank_Solver.h"


static PyObject * solveFP(PyObject *self, PyObject *args) {
//	INPUT:
	PyArrayObject *Params, *Phi,*T, *X ;
	//These correspond to the abgth values, the phi values, the time pts and the space pts//
	//  PyObject *func1, *arglist, *result;
	int N_phi, num_steps, num_nodes;
	double *abgt, *phis, *xs, *ts;
//	TO RETURN:
	PyArrayObject *A;
//	int a_dims[3];
	npy_intp a_dims[3];
	double *** Fs;
//	printf("Entered module\n");

	/* arguments: constraints:*/
	if (!PyArg_ParseTuple(args, "O!O!O!O!:solveFP", &PyArray_Type, &Params,
													&PyArray_Type, &Phi,
													&PyArray_Type, &T,
													&PyArray_Type, &X) ) {
		printf("PyArg failed \n");
		return NULL;
	}

	NDIM_CHECK(Params, 1); TYPE_CHECK(Params, PyArray_DOUBLE);
	NDIM_CHECK(Phi, 1); TYPE_CHECK(Phi, PyArray_DOUBLE);
	NDIM_CHECK(T, 1); TYPE_CHECK(T, PyArray_DOUBLE);
	NDIM_CHECK(X, 1); TYPE_CHECK(X, PyArray_DOUBLE);

	N_phi     = (int)Phi->dimensions[0];  //
	num_steps = (int)T->dimensions[0];  //
	num_nodes = (int)X->dimensions[0];  //

	abgt = (double *) Params->data;
	phis = (double *) Phi->data;
	ts =   (double *) T->data;
	xs =   (double *) X->data;

	/* create return array: */
	a_dims[0] = N_phi;
	a_dims[1] = num_steps;
	a_dims[2] = num_nodes;

	//	A = (PyArrayObject *) PyArray_SimpleNew(3, a_dims, PyArray_DOUBLE);
	A = (PyArrayObject *) PyArray_ZEROS(3, a_dims, PyArray_DOUBLE, 0);
	if (A == NULL) {
		printf("creating %d x %d  x %d array failed\n", (int)a_dims[0], (int)a_dims[1], (int)a_dims[2]);
		return NULL; /* PyArray_FromDims raises an exception */
	}

//  Allocate 3D Array (contiguously!)!!!
	double*contiguousAllocationPtr = malloc(N_phi * num_steps * num_nodes * sizeof(double));
	Fs = malloc(N_phi * sizeof(double**));
	for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx) {
		Fs[phi_idx] = malloc(num_steps* sizeof(double*) );
		for (int t_idx = 0; t_idx < num_steps; ++t_idx)
			Fs[phi_idx][t_idx] = contiguousAllocationPtr + (phi_idx * num_steps*num_nodes ) + (t_idx*num_nodes);
	}

//	Fs = (double***)malloc(N_phi * sizeof(double**));
//	for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx) {
//			Fs[phi_idx] = (double **) malloc(num_steps* sizeof(double*) );
//			for (int t_idx = 0; t_idx < num_steps; ++t_idx)
//				Fs[phi_idx][t_idx] = A->data + (phi_idx * num_steps*num_nodes ) + (t_idx*num_nodes);
//	}

	if (NULL == Fs){
		printf("Ooops Fs was not allocated");
	}

//	Fs = malloc(N_phi * sizeof(double**));
//	if (!PyArray_AsCArray(&A, Fs, a_dims, 3, PyArray_DOUBLE))
//			printf("Failed to Convert NumPy to C");

//	printf("converted N to C\n");

	_solveFP(abgt,phis, N_phi,
					ts, num_steps, xs, num_nodes, Fs);

	//Assign temp array to A->data
	double * a_ijk;
	for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx) {
		for (int t_idx = 0; t_idx < num_steps; ++t_idx) {
			for (int x_idx = 0; x_idx < num_nodes; ++x_idx) {
				//det pointer to A->data (using A->strides!! (very useful)
				a_ijk = (double *)(A->data + phi_idx*A->strides[0]
				                           + t_idx*A->strides[1]
				                           + x_idx*A->strides[2]);
				//bind to ptr and inscribe:
				*a_ijk = Fs[phi_idx][t_idx][x_idx];
			}
		}
	}
//	CLEAN UP AFTER YOURSELF!!!

	free(contiguousAllocationPtr);
	for (int phi_idx = 0; phi_idx < N_phi; ++phi_idx)
		free(Fs[phi_idx]);
	free(Fs);
//	A->data = Fs; //this would be also nice!
	return PyArray_Return(A);
}


static PyObject * simulateSDE(PyObject *self, PyObject *args) {
//	INPUT:
	PyArrayObject *Params ; //These correspond to the abgth values
	int N_spikes;
	double dt;

	double * abgt;
//	TO RETURN:
	PyArrayObject *A;
	npy_intp a_dims[1];
	double * Ss; //spike times

	/* arguments:*/
	if (!PyArg_ParseTuple(args, "O!id:simulateSDE", &PyArray_Type, &Params,
												&N_spikes,
												&dt) ) {
		printf("PyArg failed \n");
		return NULL;
	}

	NDIM_CHECK(Params, 1); TYPE_CHECK(Params, PyArray_DOUBLE);

	//assign data:
	abgt = (double *) Params->data;

	/* create return array: */
	a_dims[0] = N_spikes;
	A = (PyArrayObject *) PyArray_ZEROS(1, a_dims, PyArray_DOUBLE, 0);
	if (A == NULL) {
		printf("creating %d array failed\n", (int)a_dims[0]);
		return NULL; /* PyArray_FromDims raises an exception */
	}

//  Allocate 1D Array (contiguously!)!!!
//	double * contiguousAllocationPtr = malloc(N_phi * num_steps * num_nodes * sizeof(double));
	Ss = malloc(N_spikes * sizeof(double));

	if (NULL == Ss){
		printf("Ooops Ss was not allocated");
	}

	_simulateSDE(abgt, N_spikes, dt,
				Ss);

	//Assign temp array to A->data
	double * a_k;
	for (int t_idx = 0; t_idx < N_spikes; ++t_idx) {
				//det pointer to A->data (using A->strides!! (very useful)
				a_k = (double *)(A->data + t_idx*A->strides[0] );
				//bind to ptr and inscribe:
				*a_k = Ss[t_idx];
	}
	free(Ss);
//	A->data = Ss; //this would be also nice!
	return PyArray_Return(A);
}


static PyObject * FortetRHS(PyObject *self, PyObject *args) {
////	INPUT:
	PyArrayObject *Params, *Ts, *Is ; //These correspond to the abgth values, the phi values, the time pts and the space pts//
//	//  PyObject *func1, *arglist, *result;
//	int N_phi, num_steps, num_nodes;
	double *abgtphi, *ts, *is;
////	TO RETURN:
	PyArrayObject *RHS;
	npy_intp a_dims[1];

	int num_steps, num_intervals;
//
	/* arguments:*/
	if (!PyArg_ParseTuple(args, "O!O!O!:FortetRHS", &PyArray_Type, &Params,
												&PyArray_Type, &Ts,
												&PyArray_Type, &Is) ){
		printf("PyArg failed \n");
		return NULL;
	}
	NDIM_CHECK(Params, 1); TYPE_CHECK(Params, PyArray_DOUBLE);
	NDIM_CHECK(Ts, 1); TYPE_CHECK(Ts, PyArray_DOUBLE);
	NDIM_CHECK(Is, 1); TYPE_CHECK(Is, PyArray_DOUBLE);

	num_steps = (int)Ts->dimensions[0];  //
	num_intervals = (int)Is->dimensions[0];  //

	abgtphi = (double *) Params->data;
	ts =   (double *) Ts->data;
	is =   (double *) Is->data;

	/* create return array: */
	a_dims[0] = num_steps;

	RHS = (PyArrayObject *) PyArray_SimpleNew(1, a_dims, PyArray_DOUBLE);
	if (RHS == NULL) {
		printf("creating %d array failed\n", (int)a_dims[0]);
		return NULL; /* PyArray_FromDims raises an exception */
	}

	//temp solution
	double*rhs = malloc(num_steps * sizeof(double));

	_fortetRHS(ts, num_steps, is, num_intervals,
				abgtphi,
				rhs);

	//Assign temp array to A->data
	double * a_i;
	for (int t_idx = 0; t_idx < num_steps; ++t_idx) {
		//det pointer to A->data (using A->strides!! (very useful)
				a_i = (double *)(RHS->data + t_idx*RHS->strides[0]);
				//bind to ptr and inscribe:
				*a_i = rhs[t_idx];
	}
//	CLEAN UP AFTER YOURSELF!!!
	free(rhs);

	//	A->data = rhs; //this would be also nice!
	return PyArray_Return(RHS);
}


static PyObject * FortetError(PyObject *self, PyObject *args) {
////	INPUT:
	PyArrayObject *Params, *Ts, *Is ; //These correspond to the abgth values, the phi values, the time pts and the space pts//
//	//  PyObject *func1, *arglist, *result;
//	int N_phi, num_steps, num_nodes;
	double *abgtphi, *ts, *is;
////	TO RETURN:
	PyArrayObject *Res;
	npy_intp a_dims[1];

	int num_steps, num_intervals;
//
	/* arguments:*/
	if (!PyArg_ParseTuple(args, "O!O!O!:FortetError", &PyArray_Type, &Params,
												&PyArray_Type, &Ts,
												&PyArray_Type, &Is) ){
		printf("PyArg failed \n");
		return NULL;
	}
	NDIM_CHECK(Params, 1); TYPE_CHECK(Params, PyArray_DOUBLE);
	NDIM_CHECK(Ts, 1); TYPE_CHECK(Ts, PyArray_DOUBLE);
	NDIM_CHECK(Is, 1); TYPE_CHECK(Is, PyArray_DOUBLE);

	num_steps = (int)Ts->dimensions[0];  //
	num_intervals = (int)Is->dimensions[0];  //

	abgtphi = (double *) Params->data;
	ts =   (double *) Ts->data;
	is =   (double *) Is->data;

	/* create return array: */
	a_dims[0] = num_steps;

	Res = (PyArrayObject *) PyArray_SimpleNew(1, a_dims, PyArray_DOUBLE);
	if (Res == NULL) {
		printf("creating %d array failed\n", (int)a_dims[0]);
		return NULL; /* PyArray_FromDims raises an exception */
	}

	//temp solution
	double*err = malloc(num_steps * sizeof(double));

	_fortetError(ts, num_steps, is, num_intervals,
				abgtphi,
				err);

	//Assign temp array to A->data
	double * a_i;
	for (int t_idx = 0; t_idx < num_steps; ++t_idx) {
		//det pointer to A->data (using A->strides!! (very useful)
				a_i = (double *)(Res->data + t_idx*Res->strides[0]);
				//bind to ptr and inscribe:
				*a_i = err[t_idx];
	}
//	CLEAN UP AFTER YOURSELF!!!
	free(err);

	//	A->data = rhs; //this would be also nice!
	return PyArray_Return(Res);
}


/* p solve */
static PyObject * solve_p(PyObject *self, PyObject *args) {
	//	INPUT:

}

/* f solve */
static PyObject * solve_f(PyObject *self, PyObject *args) {
	//	INPUT:
	// The Python (array) 'objects' passed in:
	PyArrayObject *Params, *A,*T, *X ;
	// These correspond to the mutausigma values, the alphas values, the time pts and the space pts //
	// PyObject *func1, *arglist, *result;
	int num_steps, num_nodes;
	double *mutausigma, *alphas, *ts, *xs;

	// WORKING VARIABLES:
	//Raw data for the return array:
	double (** fs);

	//OUTPUT:
	//RETURN Obj:
	const int N_dims_output = 2;
	npy_intp a_dims[N_dims_output];
	PyArrayObject * returnObj;

//	printf(" ext_fpc:solve_f  Entered module\n");
	/* arguments: constraints:*/
	if (!PyArg_ParseTuple(args, "O!O!O!O!:solve_f",
							&PyArray_Type, &Params,
							&PyArray_Type, &A,
							&PyArray_Type, &T,
							&PyArray_Type, &X) ) {
		printf("PyArg failed \n");
		return NULL;
	}

	NDIM_CHECK(Params, 1); TYPE_CHECK(Params, PyArray_DOUBLE);
	NDIM_CHECK(A, 1); TYPE_CHECK(A, PyArray_DOUBLE);
	NDIM_CHECK(T, 1); TYPE_CHECK(T, PyArray_DOUBLE);
	NDIM_CHECK(X, 1); TYPE_CHECK(X, PyArray_DOUBLE);

	num_steps = (int)T->dimensions[0];  //
	num_nodes = (int)X->dimensions[0];  //

	mutausigma = (double *) Params->data;
	alphas = (double * )A->data;
	ts =   (double *) T->data;
	xs =   (double *) X->data;

	//  Allocate a 2D Array  (contiguously!)!!!
	fs = malloc(num_steps * sizeof(double *));
	if (NULL == fs){
		printf("Ooops fs was not allocated");
		return NULL;
	}
	fs[0] = malloc(num_steps*num_nodes * sizeof(double));
	if (NULL == fs[0]){
		printf("Ooops fs[0] was not allocated");
		return NULL;
	}
	for(size_t idx = 1; idx < num_steps; ++idx) {
		fs[idx] = fs[0] + idx * num_nodes;
	}

//	//MAIN CALL:
	_solve_f(mutausigma, alphas,
			ts, num_steps,
			xs, num_nodes,
			 fs);



	//now return to Python:
	//remember to transpose f:
	a_dims[0] = num_nodes;
	a_dims[1] = num_steps;

	returnObj = (PyArrayObject *) PyArray_ZEROS(N_dims_output, a_dims, PyArray_DOUBLE, 0);
	if(NULL == returnObj){
		printf("creating %d  x %d array failed\n",
				(int)a_dims[0], (int)a_dims[1] );
		return NULL; /* PyArray_FromDims raises an exception */
	}

	//Assign working memory array to returnObj->data
	//You can rearrange the array here (permute dimensions),
	// so that the compute (working) array and the return array may be permuted vis-a-vis each other as the API requires
	// We do this here to fit the (wrong) way it was originally done in Python vs[x_idx, t_idx];
	double * f_jk;
	for (int t_idx = 0; t_idx < num_steps; ++t_idx) {
		for (int x_idx = 0; x_idx < num_nodes; ++x_idx) {
		//get pointer to A->data (using A->strides!! (very useful)
			f_jk = (double *)(returnObj->data
									+ x_idx*returnObj->strides[0]
									+ t_idx*returnObj->strides[1]);
			*f_jk = fs[t_idx][x_idx];
		}
	}


//CLEAN UP AFTER YOURSELF!!! (for every malloc there must be a free)
	free( (void *) fs[0] );
	free( (void *) fs );

	//return
	return PyArray_Return(returnObj);
}
//

/* doc strings: */
static char FokkerPlank_solve_doc[] = \
  "Fs = solveFP(abgt, phis, ts, xs) //Calculates the F-P solution corresponding to the structural parameters in abgt on a grid of phis, ts, xs";
static char SimulateSDE_solve_doc[] = \
  "Ss = simulateSDE(abgt, N_spikes, dt) // Calculate N_spikes from a sinusoidal LIF";
static char FortetRHS_solve_doc[] = \
  "rhs = FortetRHS(abgthphi, ts, Is) // Right hand side of the Fortet Equation for fixed phi";
static char FortetError_solve_doc[] = \
  "rhs = FortetError(abgthphi, ts, Is) // The sample error of the Fortet Equation for fixed phi";
static char solve_f_doc[] =\
  "fs = solve_f(mutausigma, alphas, ts, xs) // calculate the f-p density soln corresponding to params + control + ts + xs";

static char module_doc[] = \
  "module ext_fpc:\n\
   Fs = solveFP(abgt, phis, ts, xs) //Calculates the F-P solution corresponding to the structural parameters in abgt on a grid of phis, ts, xs \n\
   Ss = simulateSDE(abgt, N_spikes, dt) // Calculate N_spikes from a sinusoidal LIF \n\
   rhs = fortetRHS(ts, Is) // Calculate the Right-hand side of the fortet equation  \n\
   fs = solve_f(mutausigma, alphas, ts, xs) // calculate the f-p density\
  ";

/* 
   The method table must always be present - it lists the functions that should be callable from Python:
*/
static PyMethodDef ext_fpc_methods[] = {
  {"solveFP",    /* name of func when called from Python */
    solveFP,      /* corresponding C function */
    METH_VARARGS,   /* ordinary (not keyword) arguments */
    FokkerPlank_solve_doc}, /* doc string for solve function */
  {"simulateSDE",    /* name of func when called from Python */
    simulateSDE,      /* corresponding C function */
    METH_VARARGS,   /* ordinary (not keyword) arguments */
    SimulateSDE_solve_doc}, /* doc string for solve function */
  {"FortetRHS",    /* name of func when called from Python */
	FortetRHS,      /* corresponding C function */
	METH_VARARGS,   /* ordinary (not keyword) arguments */
	FortetRHS_solve_doc}, /* doc string for solve function */
  {"FortetError",    /* name of func when called from Python */
	FortetError,      /* corresponding C function */
	METH_VARARGS,   /* ordinary (not keyword) arguments */
	FortetError_solve_doc}, /* doc string for solve function */
	{"solve_f",
	solve_f,
	METH_VARARGS,
	solve_f_doc},
   // in general {"python name", cname, METH_VARARGS (usually), doc string if you want it}
   {NULL, NULL}     /* required ending of the method table */
};

void initext_fpc()
{
  /* Assign the name of the module and the name of the
     method table and (optionally) a module doc string:
  */
  Py_InitModule3("ext_fpc", ext_fpc_methods, module_doc);
  /* without module doc string: 
  Py_InitModule ("ext_fpc", ext_fpc_methods); */

  import_array();   /* required NumPy initialization */
}
