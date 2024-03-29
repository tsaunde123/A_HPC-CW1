/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <mpi.h>
#include <string.h>

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define MASTER          0
#define NTYPES          1  /* the number of intrinsic types in our derived type */


/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int rank);

int calc_nrows_from_rank(int rank, int size, int numRows);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, t_speed* halo_cells, int* halo_obs, int local_nrows, int local_ncols, int size, int rank, int halo_local_nrows, int halo_local_ncols, int nlr_nrows, t_speed* halo_temp, MPI_Status status, int top,
 int bottom, MPI_Datatype MPI_cell_type);
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, t_speed* halo_cells, int* halo_obs, int local_nrows);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells, t_speed* halo_cells, int local_ncols, int local_nrows, int nlr_nrows, int halo_local_nrows, int halo_local_ncols, int rank, int size, t_speed* halo_temp);
int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int local_nrows, int local_ncols, int* halo_obs,
            t_speed* halo_cells, int rank, int size, int nlr_nrows, t_speed* halo_temp);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int local_nrows, int local_ncols, int* halo_obs,
              t_speed* halo_temp, t_speed* halo_cells);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* Halo exchange method */
void halo_ex(t_speed* halo_cells, int halo_local_ncols, int halo_local_nrows, MPI_Status status, int local_nrows, int local_ncols, int top, int bottom, MPI_Datatype MPI_cell_type);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float final_av_velocity(const t_param params, t_speed* cells, int* obstacles);
float av_velocity(const t_param params, t_speed* cells, int* obstacles, int local_nrows, int local_ncols, t_speed* halo_cells, int rank,
                  int size, MPI_Status status, int* halo_obs);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char* file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  t_speed* cells     = NULL;    /* grid containing fluid densities */
  t_speed* tmp_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */
  int rank;                     /* 'rank' of process among it's cohort */
  int size;                     /* size of cohort, i.e. num processes started */
  MPI_Status status;     /* struct used by MPI_Recv */
  int local_nrows;       /* number of rows apportioned to this rank */
  int local_ncols;       /* number of columns apportioned to this rank */
  int remote_nrows;      /* number of columns apportioned to a remote rank */
  int nlr_nrows;         /* so last rank know num rows in other ranks */
  t_speed* sendbuf;       /* buffer to hold values to send */
  t_speed* recvbuf;       /* buffer to hold received values */
  t_speed* printbuf;      /* buffer to hold values for printing */


  /* parse the command line */
  if (argc != 3)
  {
    usage(argv[0]);
  }
  else
  {
    paramfile = argv[1];
    obstaclefile = argv[2];
  }


  MPI_Init( &argc, &argv );
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  MPI_Datatype types[NTYPES] = {MPI_FLOAT}; //NTYPES=1 for now
  int blocklengths[NTYPES] = {NSPEEDS};
  MPI_Datatype MPI_cell_type;
  MPI_Aint     offsets[NTYPES];
  offsets[0] = offsetof(t_speed, speeds);

  MPI_Type_create_struct(NTYPES, blocklengths, offsets, types, &MPI_cell_type);
  MPI_Type_commit(&MPI_cell_type);

  MPI_Comm_rank( MPI_COMM_WORLD, &rank );


  /* initialise our data structures and load values from file */
  initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, rank);

  int top = (rank + 1) % size;
  int bottom = (rank == MASTER) ? (rank + size - 1) : (rank - 1);
  local_nrows = calc_nrows_from_rank(rank, size, params.ny);
  local_ncols = params.nx;
  int extra_local_nrows = calc_nrows_from_rank(MASTER, size, params.ny); //master will always be first to have extra row
  int not_extra_local_nrows;
  int tag = 0;
  int rest = params.ny % size;

  sendbuf = (t_speed*)malloc(sizeof(t_speed) * local_ncols);
  recvbuf = (t_speed*)malloc(sizeof(t_speed) * local_ncols);
  /* The last rank has the most columns apportioned.
     printbuf must be big enough to hold this number */
  // remote_nrows = calc_ncols_from_rank(size-1, size, params.ny);
  //nlr_nrows = calc_ncols_from_rank(MASTER, size, params.ny);

  t_speed* halo_cells = (t_speed*)malloc(sizeof(t_speed) * (local_nrows+2) * local_ncols);
  int halo_local_nrows = local_nrows + 2;
  int halo_local_ncols = local_ncols;
  int* halo_obs = (int*)malloc(sizeof(int) * local_nrows * local_ncols);
  t_speed* halo_temp = (t_speed*)malloc(sizeof(t_speed) * local_nrows * local_ncols);

  t_speed* sendcbuf = (t_speed*)malloc(sizeof(t_speed) * extra_local_nrows * local_ncols);
  t_speed* recvcbuf = (t_speed*)malloc(sizeof(t_speed) * extra_local_nrows * local_ncols);
  int* sendobuf = (int*)malloc(sizeof(int) * extra_local_nrows * local_ncols);
  int* recvobuf = (int*)malloc(sizeof(int) * extra_local_nrows * local_ncols);
  //t_speed* sendtbuf = (t_speed*)malloc(sizeof(t_speed) * local_nrows * local_ncols);
  //t_speed* recvtbuf = (t_speed*)malloc(sizeof(t_speed) * local_nrows * local_ncols);

  if(rank == MASTER){
    for(int jj = 0; jj < local_ncols*local_nrows; jj++){
      halo_cells[jj+(1*local_ncols)] = cells[jj + (rank*(local_nrows*local_ncols))];
      halo_obs[jj] = obstacles[jj + (rank*(local_nrows*local_ncols))];
    }
    //memcpy( void* dest, const void* src, std::size_t count );
    for(int dest = 1; dest < size; dest++){
      if(dest < rest){ //if dest has extra row
        // t_speed* sendbuf = (t_speed*)malloc(sizeof(t_speed) * remote_nrows * local_ncols);
        // int* sendobuf = (int*)malloc(sizeof(int) * remote_nrows * local_ncols);
        for(int jj = 0; jj < extra_local_nrows * local_ncols; jj++){
          sendcbuf[jj] = cells[jj + (dest*(extra_local_nrows*local_ncols))];
          sendobuf[jj] = obstacles[jj + (dest*(extra_local_nrows*local_ncols))];
        }
        MPI_Send(sendcbuf, extra_local_nrows*local_ncols, MPI_cell_type, dest, tag, MPI_COMM_WORLD);
        MPI_Send(sendobuf, extra_local_nrows*local_ncols, MPI_INT, dest, tag, MPI_COMM_WORLD);
      } else{
        not_extra_local_nrows = calc_nrows_from_rank(dest, size, params.ny);
        // t_speed* sendcbuf = (t_speed*)malloc(sizeof(t_speed) * local_nrows * local_ncols);
        // int* sendobuf = (int*)malloc(sizeof(int) * local_nrows * local_ncols);
        for(int jj = 0; jj < not_extra_local_nrows * local_ncols; jj++){
          sendcbuf[jj] = cells[jj + ((rest*(extra_local_nrows*local_ncols)) + (dest-rest)*(not_extra_local_nrows*local_ncols))];
          sendobuf[jj] = obstacles[jj + ((rest*(extra_local_nrows*local_ncols)) + (dest-rest)*(not_extra_local_nrows*local_ncols))];
        }
        MPI_Send(sendcbuf, not_extra_local_nrows*local_ncols, MPI_cell_type, dest, tag, MPI_COMM_WORLD);
        MPI_Send(sendobuf, not_extra_local_nrows*local_ncols, MPI_INT, dest, tag, MPI_COMM_WORLD);
      }
    }
  } else {
    if(rank < rest){
      // t_speed* rcvcbuf = (t_speed*)malloc(sizeof(t_speed) * remote_nrows * local_ncols);
      // int* rcvobuf = (int*)malloc(sizeof(int) * remote_nrows * local_ncols);
      MPI_Recv(recvcbuf, extra_local_nrows*local_ncols, MPI_cell_type, MASTER, tag, MPI_COMM_WORLD, &status);
      for(int jj = 0; jj < local_nrows * halo_local_ncols; jj++){
        halo_cells[jj + halo_local_ncols*1] = recvcbuf[jj];
      }
      MPI_Recv(recvobuf, extra_local_nrows*local_ncols, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
      for(int jj = 0; jj < local_nrows*local_ncols; jj++){
        halo_obs[jj] = recvobuf[jj];
      }
    } else {
      // t_speed* rcvcbuf = (t_speed*)malloc(sizeof(t_speed) * local_nrows * local_ncols);
      // int* rcvobuf = (int*)malloc(sizeof(int) * local_nrows * local_ncols);
      MPI_Recv(recvcbuf,local_nrows*local_ncols, MPI_cell_type, MASTER, tag, MPI_COMM_WORLD, &status);
      for(int jj = 0; jj < local_nrows * halo_local_ncols; jj++){
        halo_cells[jj + halo_local_ncols*1] = recvcbuf[jj];
      }
      MPI_Recv(recvobuf, local_nrows*halo_local_ncols, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
      for(int jj = 0; jj < local_nrows*local_ncols; jj++){
        halo_obs[jj] = recvobuf[jj];
      }
    }
  }

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  for (int tt = 0; tt < params.maxIters; tt++)
  {
    timestep(params, cells, tmp_cells, obstacles, halo_cells, halo_obs, local_nrows, local_ncols, size, rank, halo_local_nrows, halo_local_ncols, nlr_nrows, halo_temp, status, top, bottom, MPI_cell_type);
    //av_vels[tt] = av_velocity(params, cells, obstacles);
    av_vels[tt] = av_velocity(params, cells, obstacles, local_nrows, local_ncols, halo_cells, rank, size, status, halo_obs);
#ifdef DEBUG
    printf("==timestep: %d==\n", tt);
    printf("av velocity: %.12E\n", av_vels[tt]);
    printf("tot density: %.12E\n", total_density(params, cells));
#endif
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  if(rank != MASTER){
    if(rank < rest){
      for(int jj = 0; jj < extra_local_nrows * local_ncols; jj++){
        sendcbuf[jj] = halo_cells[jj + 1*local_ncols];
        sendobuf[jj] = halo_obs[jj + 1*local_ncols];
      }
      MPI_Send(sendcbuf, extra_local_nrows*local_ncols, MPI_cell_type, MASTER, tag, MPI_COMM_WORLD);
      MPI_Send(sendobuf, extra_local_nrows*local_ncols, MPI_INT, MASTER, tag, MPI_COMM_WORLD);
    } else {
      for(int jj = 0; jj < local_nrows * local_ncols; jj++){
        sendcbuf[jj] = halo_cells[jj + 1*local_ncols];
        sendobuf[jj] = halo_obs[jj + 1*local_ncols];
      }
      MPI_Send(sendcbuf, local_nrows*local_ncols, MPI_cell_type, MASTER, tag, MPI_COMM_WORLD);
      MPI_Send(sendobuf, local_nrows*local_ncols, MPI_INT, MASTER, tag, MPI_COMM_WORLD);

    }
  } else {
    for(int jj = 0; jj < local_ncols*local_nrows; jj++){
      cells[jj] = halo_cells[jj + 1*local_ncols];
    }
    for(int source = 1; source < size; source++){
      if(source < rest){
        MPI_Recv(recvcbuf, extra_local_nrows*local_ncols, MPI_cell_type, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(recvobuf, extra_local_nrows*local_ncols, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        for(int jj = 0; jj < extra_local_nrows*local_ncols; jj++){
          cells[jj + (source*(local_nrows*local_ncols))] = recvcbuf[jj];
          obstacles[jj + (source*(local_nrows*local_ncols))] = recvobuf[jj];
        }
      } else {
        not_extra_local_nrows = calc_nrows_from_rank(source, size, params.ny);
        MPI_Recv(recvcbuf, not_extra_local_nrows*local_ncols, MPI_cell_type, source, tag, MPI_COMM_WORLD, &status);
        MPI_Recv(recvobuf, not_extra_local_nrows*local_ncols, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
        for(int jj = 0; jj < not_extra_local_nrows*local_ncols; jj++){
          cells[jj + ((rest*extra_local_nrows*local_ncols) + (source-rest)*(not_extra_local_nrows*local_ncols))] = recvcbuf[jj];
          obstacles[jj + ((rest*extra_local_nrows*local_ncols) + (source-rest)*(not_extra_local_nrows*local_ncols))] = recvobuf[jj];
        }
      }
    }
  }


  /* write final values and free memory */
  if(rank == MASTER){
    printf("==done==\n");
    printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
    write_values(params, cells, obstacles, av_vels);
  }
  finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

  MPI_Finalize();

  return EXIT_SUCCESS;
}

int timestep(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, t_speed* halo_cells, int* halo_obs, int local_nrows, int local_ncols, int size, int rank, int halo_local_nrows, int halo_local_ncols, int nlr_nrows, t_speed* halo_temp, MPI_Status status, int top,
 int bottom, MPI_Datatype MPI_cell_type)
{
  if(calc_nrows_from_rank(size-1, size, params.ny) == 1){
    if(rank == size-2){
      accelerate_flow(params, cells, obstacles, halo_cells, halo_obs, local_nrows);
    }
  } else {
    if(rank == size-1){
      accelerate_flow(params, cells, obstacles, halo_cells, halo_obs, local_nrows);
    }
  }

  halo_ex(halo_cells, halo_local_ncols, halo_local_nrows, status, local_nrows, local_ncols, top, bottom, MPI_cell_type);

  //accelerate_flow(params, cells, obstacles, halo_cells, halo_obs, local_nrows);
  propagate(params, cells, tmp_cells, halo_cells, local_ncols, local_nrows, nlr_nrows, halo_local_nrows, halo_local_ncols, rank, size, halo_temp);
  rebound(params, cells, tmp_cells, obstacles, local_nrows, local_ncols, halo_obs, halo_cells, rank, size, nlr_nrows, halo_temp);
  collision(params, cells, tmp_cells, obstacles, local_nrows, local_ncols, halo_obs, halo_temp, halo_cells);
  return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles, t_speed* halo_cells, int* halo_obs, int local_nrows)
{
  /* compute weighting factors */
  float w1 = params.density * params.accel / 9.f;
  float w2 = params.density * params.accel / 36.f;

  /* modify the 2nd row of the grid */
  int h_jj = local_nrows - 1;
  int o_jj = local_nrows - 2;
  int jj = params.ny-2;

  /*for (int ii = 0; ii < params.nx; ii++)
    {
      // if the cell is not occupied and
      //     ** we don't send a negative density
      if (!obstacles[ii + jj*params.nx]
          && (cells[ii + jj*params.nx].speeds[3] - w1) > 0.f
          && (cells[ii + jj*params.nx].speeds[6] - w2) > 0.f
          && (cells[ii + jj*params.nx].speeds[7] - w2) > 0.f)
      {
        // increase 'east-side' densities
        cells[ii + jj*params.nx].speeds[1] += w1;
        cells[ii + jj*params.nx].speeds[5] += w2;
        cells[ii + jj*params.nx].speeds[8] += w2;
        // decrease 'west-side' densities
        cells[ii + jj*params.nx].speeds[3] -= w1;
        cells[ii + jj*params.nx].speeds[6] -= w2;
        cells[ii + jj*params.nx].speeds[7] -= w2;
      }
    }*/

  for (int ii = 0; ii < params.nx; ii++)
  {
    // if the cell is not occupied and
    // we don't send a negative density
    if (!halo_obs[ii + o_jj*params.nx]
        && (halo_cells[ii + h_jj*params.nx].speeds[3] - w1) > 0.f
        && (halo_cells[ii + h_jj*params.nx].speeds[6] - w2) > 0.f
        && (halo_cells[ii + h_jj*params.nx].speeds[7] - w2) > 0.f)
    {
      // increase 'east-side' densities
      halo_cells[ii + h_jj*params.nx].speeds[1] += w1;
      halo_cells[ii + h_jj*params.nx].speeds[5] += w2;
      halo_cells[ii + h_jj*params.nx].speeds[8] += w2;
      // decrease 'west-side' densities
      halo_cells[ii + h_jj*params.nx].speeds[3] -= w1;
      halo_cells[ii + h_jj*params.nx].speeds[6] -= w2;
      halo_cells[ii + h_jj*params.nx].speeds[7] -= w2;
    }
  }



  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells, t_speed* halo_cells, int local_ncols, int local_nrows, int nlr_nrows,
              int halo_local_nrows, int halo_local_ncols, int rank, int size, t_speed* halo_temp)
{
  /* loop over _all_ cells */
  /*for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      // determine indices of axis-direction neighbours
      // respecting periodic boundary conditions (wrap around)
      int y_n = (jj + 1) % params.ny;
      int x_e = (ii + 1) % params.nx;
      int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);
      int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
      // propagate densities from neighbouring cells, following
      // appropriate directions of travel and writing into
      // scratch space grid
      tmp_cells[ii + jj*params.nx].speeds[0] = cells[ii + jj*params.nx].speeds[0]; // central cell, no movement
      tmp_cells[ii + jj*params.nx].speeds[1] = cells[x_w + jj*params.nx].speeds[1]; // east
      tmp_cells[ii + jj*params.nx].speeds[2] = cells[ii + y_s*params.nx].speeds[2]; // north
      tmp_cells[ii + jj*params.nx].speeds[3] = cells[x_e + jj*params.nx].speeds[3]; // west
      tmp_cells[ii + jj*params.nx].speeds[4] = cells[ii + y_n*params.nx].speeds[4]; // south
      tmp_cells[ii + jj*params.nx].speeds[5] = cells[x_w + y_s*params.nx].speeds[5]; // north-east
      tmp_cells[ii + jj*params.nx].speeds[6] = cells[x_e + y_s*params.nx].speeds[6]; // north-west
      tmp_cells[ii + jj*params.nx].speeds[7] = cells[x_e + y_n*params.nx].speeds[7]; // south-west
      tmp_cells[ii + jj*params.nx].speeds[8] = cells[x_w + y_n*params.nx].speeds[8]; // south-east/
    }
  }*/
  for (int jj = 0; jj < local_nrows; jj++)
  {
    for (int ii = 0; ii < local_ncols; ii++)
    {
      int y_n = (jj+1) + 1;
      int x_e = (ii + 1) % halo_local_ncols; //((ii+1) + 1);
      int y_s = (jj+1) - 1;
      int x_w = (ii == 0) ? (ii + halo_local_ncols - 1) : (ii - 1); //((ii+1) - 1);

      halo_temp[(ii + jj*params.nx)].speeds[0] = halo_cells[ii + (jj+1)*halo_local_ncols].speeds[0]; /* central cell, no movement */
      halo_temp[(ii + jj*params.nx)].speeds[1] = halo_cells[x_w + (jj+1)*local_ncols].speeds[1]; /* east */
      halo_temp[(ii + jj*params.nx)].speeds[2] = halo_cells[ii + y_s*local_ncols].speeds[2]; /* north */
      halo_temp[(ii + jj*params.nx)].speeds[3] = halo_cells[x_e + (jj+1)*local_ncols].speeds[3]; /* west */
      halo_temp[(ii + jj*params.nx)].speeds[4] = halo_cells[ii + y_n*local_ncols].speeds[4]; /* south */
      halo_temp[(ii + jj*params.nx)].speeds[5] = halo_cells[x_w + y_s*local_ncols].speeds[5]; /* north-east */
      halo_temp[(ii + jj*params.nx)].speeds[6] = halo_cells[x_e + y_s*local_ncols].speeds[6]; /* north-west */
      halo_temp[(ii + jj*params.nx)].speeds[7] = halo_cells[x_e + y_n*local_ncols].speeds[7]; /* south-west */
      halo_temp[(ii + jj*params.nx)].speeds[8] = halo_cells[x_w + y_n*local_ncols].speeds[8]; /* south-east */
    }
  }

  return EXIT_SUCCESS;
}

int rebound(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int local_nrows, int local_ncols, int* halo_obs,
            t_speed* halo_cells, int rank, int size, int nlr_nrows, t_speed* halo_temp)
{
  /* //loop over the cells in the grid
  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < local_ncols; ii++)
    {
      // if the cell contains an obstacle
      if (obstacles[jj*params.nx + ii])
      {
        // called after propagate, so taking values from scratch space
        // mirroring, and writing into main grid
        cells[ii + jj*params.nx].speeds[1] = tmp_cells[ii + jj*params.nx].speeds[3];
        cells[ii + jj*params.nx].speeds[2] = tmp_cells[ii + jj*params.nx].speeds[4];
        cells[ii + jj*params.nx].speeds[3] = tmp_cells[ii + jj*params.nx].speeds[1];
        cells[ii + jj*params.nx].speeds[4] = tmp_cells[ii + jj*params.nx].speeds[2];
        cells[ii + jj*params.nx].speeds[5] = tmp_cells[ii + jj*params.nx].speeds[7];
        cells[ii + jj*params.nx].speeds[6] = tmp_cells[ii + jj*params.nx].speeds[8];
        cells[ii + jj*params.nx].speeds[7] = tmp_cells[ii + jj*params.nx].speeds[5];
        cells[ii + jj*params.nx].speeds[8] = tmp_cells[ii + jj*params.nx].speeds[6];
      }
    }
  }*/

  for (int jj = 0; jj < local_nrows; jj++)
  {
    for (int ii = 0; ii < local_ncols; ii++)
    {
      if (halo_obs[jj*params.nx + ii])
      {
        halo_cells[ii + (jj+1)*params.nx].speeds[1] = halo_temp[ii + jj*params.nx].speeds[3];
        halo_cells[ii + (jj+1)*params.nx].speeds[2] = halo_temp[ii + jj*params.nx].speeds[4];
        halo_cells[ii + (jj+1)*params.nx].speeds[3] = halo_temp[ii + jj*params.nx].speeds[1];
        halo_cells[ii + (jj+1)*params.nx].speeds[4] = halo_temp[ii + jj*params.nx].speeds[2];
        halo_cells[ii + (jj+1)*params.nx].speeds[5] = halo_temp[ii + jj*params.nx].speeds[7];
        halo_cells[ii + (jj+1)*params.nx].speeds[6] = halo_temp[ii + jj*params.nx].speeds[8];
        halo_cells[ii + (jj+1)*params.nx].speeds[7] = halo_temp[ii + jj*params.nx].speeds[5];
        halo_cells[ii + (jj+1)*params.nx].speeds[8] = halo_temp[ii + jj*params.nx].speeds[6];
      }
    }
  }

  return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, int local_nrows, int local_ncols, int* halo_obs,
              t_speed* halo_temp, t_speed* halo_cells)
{
  const float c_sq = 1.f / 3.f; /* square of speed of sound */
  const float w0 = 4.f / 9.f;  /* weighting factor */
  const float w1 = 1.f / 9.f;  /* weighting factor */
  const float w2 = 1.f / 36.f; /* weighting factor */

  /* loop over the cells in the grid
  ** NB the collision step is called after
  ** the propagate step and so values of interest
  ** are in the scratch-space grid */
  for (int jj = 0; jj < local_nrows; jj++)
  {
    for (int ii = 0; ii < local_ncols; ii++)
    {
      /* don't consider occupied cells */
      if (!halo_obs[ii + jj*params.nx])
      {
        /* compute local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += halo_temp[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (halo_temp[ii + jj*params.nx].speeds[1]
                      + halo_temp[ii + jj*params.nx].speeds[5]
                      + halo_temp[ii + jj*params.nx].speeds[8]
                      - (halo_temp[ii + jj*params.nx].speeds[3]
                         + halo_temp[ii + jj*params.nx].speeds[6]
                         + halo_temp[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (halo_temp[ii + jj*params.nx].speeds[2]
                      + halo_temp[ii + jj*params.nx].speeds[5]
                      + halo_temp[ii + jj*params.nx].speeds[6]
                      - (halo_temp[ii + jj*params.nx].speeds[4]
                         + halo_temp[ii + jj*params.nx].speeds[7]
                         + halo_temp[ii + jj*params.nx].speeds[8]))
                     / local_density;

        /* velocity squared */
        float u_sq = u_x * u_x + u_y * u_y;

        /* directional velocity components */
        float u[NSPEEDS];
        u[1] =   u_x;        /* east */
        u[2] =         u_y;  /* north */
        u[3] = - u_x;        /* west */
        u[4] =       - u_y;  /* south */
        u[5] =   u_x + u_y;  /* north-east */
        u[6] = - u_x + u_y;  /* north-west */
        u[7] = - u_x - u_y;  /* south-west */
        u[8] =   u_x - u_y;  /* south-east */

        /* equilibrium densities */
        float d_equ[NSPEEDS];
        /* zero velocity density: weight w0 */
        d_equ[0] = w0 * local_density
                   * (1.f - u_sq / (2.f * c_sq));
        /* axis speeds: weight w1 */
        d_equ[1] = w1 * local_density * (1.f + u[1] / c_sq
                                         + (u[1] * u[1]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[2] = w1 * local_density * (1.f + u[2] / c_sq
                                         + (u[2] * u[2]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[3] = w1 * local_density * (1.f + u[3] / c_sq
                                         + (u[3] * u[3]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[4] = w1 * local_density * (1.f + u[4] / c_sq
                                         + (u[4] * u[4]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        /* diagonal speeds: weight w2 */
        d_equ[5] = w2 * local_density * (1.f + u[5] / c_sq
                                         + (u[5] * u[5]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[6] = w2 * local_density * (1.f + u[6] / c_sq
                                         + (u[6] * u[6]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[7] = w2 * local_density * (1.f + u[7] / c_sq
                                         + (u[7] * u[7]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));
        d_equ[8] = w2 * local_density * (1.f + u[8] / c_sq
                                         + (u[8] * u[8]) / (2.f * c_sq * c_sq)
                                         - u_sq / (2.f * c_sq));

        /* relaxation step */
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          halo_cells[ii + (jj+1)*params.nx].speeds[kk] = halo_temp[ii + jj*params.nx].speeds[kk]
                                                  + params.omega
                                                  * (d_equ[kk] - halo_temp[ii + jj*params.nx].speeds[kk]);
        }
      }
    }
  }

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles, int local_nrows, int local_ncols, t_speed* halo_cells, int rank,
                  int size, MPI_Status status, int* halo_obs)
{
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */
  int recv_tot_cells;
  float recv_tot_u;

  /* initialise */
  tot_u = 0.f;

  /* loop over all non-blocked cells */
  for (int jj = 1; jj < local_nrows+1; jj++)
  {
    for (int ii = 0; ii < local_ncols; ii++)
    {
      /* ignore occupied cells */
      if (!halo_obs[ii + (jj-1)*params.nx])
      {
        /* local density total */
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += halo_cells[ii + jj*params.nx].speeds[kk];
        }

        /* x-component of velocity */
        float u_x = (halo_cells[ii + jj*params.nx].speeds[1]
                      + halo_cells[ii + jj*params.nx].speeds[5]
                      + halo_cells[ii + jj*params.nx].speeds[8]
                      - (halo_cells[ii + jj*params.nx].speeds[3]
                         + halo_cells[ii + jj*params.nx].speeds[6]
                         + halo_cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        /* compute y velocity component */
        float u_y = (halo_cells[ii + jj*params.nx].speeds[2]
                      + halo_cells[ii + jj*params.nx].speeds[5]
                      + halo_cells[ii + jj*params.nx].speeds[6]
                      - (halo_cells[ii + jj*params.nx].speeds[4]
                         + halo_cells[ii + jj*params.nx].speeds[7]
                         + halo_cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        /* accumulate the norm of x- and y- velocity components */
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        /* increase counter of inspected cells */
        ++tot_cells;
      }
    }
  }

  if(rank == MASTER){
    for(int source = 1; source < size; source++){
      MPI_Recv(&recv_tot_u, 1, MPI_FLOAT, source, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&recv_tot_cells, 1, MPI_INT, source, 0, MPI_COMM_WORLD, &status);
      tot_u += recv_tot_u;
      tot_cells += recv_tot_cells;
    }
    return tot_u / (float)tot_cells;
  } else {
    MPI_Send(&tot_u, 1, MPI_FLOAT, MASTER, 0, MPI_COMM_WORLD);
    MPI_Send(&tot_cells, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD);
  }

  /*int global_tot_cells;
  float global_tot_u;

  // Reduce all of the local sums into the global sum
  MPI_Reduce(&tot_u, &global_tot_u, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&tot_cells, &global_tot_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if(rank == MASTER){
    return global_tot_u / global_tot_cells;
  }*/

 return 0;

}

float final_av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    tot_cells = 0;
  float tot_u;

  tot_u = 0.f;

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      if (!obstacles[ii + jj*params.nx])
      {
        float local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        float u_x = (cells[ii + jj*params.nx].speeds[1]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[8]
                      - (cells[ii + jj*params.nx].speeds[3]
                         + cells[ii + jj*params.nx].speeds[6]
                         + cells[ii + jj*params.nx].speeds[7]))
                     / local_density;
        float u_y = (cells[ii + jj*params.nx].speeds[2]
                      + cells[ii + jj*params.nx].speeds[5]
                      + cells[ii + jj*params.nx].speeds[6]
                      - (cells[ii + jj*params.nx].speeds[4]
                         + cells[ii + jj*params.nx].speeds[7]
                         + cells[ii + jj*params.nx].speeds[8]))
                     / local_density;
        tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
        ++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

void halo_ex(t_speed* halo_cells, int halo_local_ncols, int halo_local_nrows, MPI_Status status, int local_nrows, int local_ncols, int top,
             int bottom, MPI_Datatype MPI_cell_type)
{

  t_speed* sendbuf = (t_speed*)malloc(sizeof(t_speed) * local_ncols);
  t_speed* recvbuf = (t_speed*)malloc(sizeof(t_speed) * local_ncols);
  //Send top, receive bottom
  for(int jj = 0; jj < halo_local_ncols; jj++){
    sendbuf[jj] = halo_cells[jj + (halo_local_ncols*local_nrows)];
  }
  MPI_Sendrecv(sendbuf, halo_local_ncols, MPI_cell_type, top, 0,
                 recvbuf, halo_local_ncols, MPI_cell_type, bottom, 0,
                 MPI_COMM_WORLD, &status);
  for(int jj = 0; jj < halo_local_ncols; jj++){
    halo_cells[jj] = recvbuf[jj];
  }
  //Send bottom, receive top
  for(int jj = 0; jj < halo_local_ncols; jj++){
    sendbuf[jj] = halo_cells[jj + (halo_local_ncols*1)];
  }
  MPI_Sendrecv(sendbuf, halo_local_ncols, MPI_cell_type, bottom, 0,
                 recvbuf, halo_local_ncols, MPI_cell_type, top, 0,
                 MPI_COMM_WORLD, &status);
  for(int jj = 0; jj < halo_local_ncols; jj++){
    halo_cells[jj + (halo_local_ncols*(local_nrows+1))] = recvbuf[jj];
  }
}

int initialise(const char* paramfile, const char* obstaclefile,
               t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr, int rank)
{
  char   message[1024];  /* message buffer */
  FILE*   fp;            /* file pointer */
  int    xx, yy;         /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */
  int    retval;         /* to hold return value for checking */

  /* open the parameter file */
  fp = fopen(paramfile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input parameter file: %s", paramfile);
    die(message, __LINE__, __FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp, "%d\n", &(params->nx));

  if (retval != 1) die("could not read param file: nx", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->ny));

  if (retval != 1) die("could not read param file: ny", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->maxIters));

  if (retval != 1) die("could not read param file: maxIters", __LINE__, __FILE__);

  retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

  if (retval != 1) die("could not read param file: reynolds_dim", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->density));

  if (retval != 1) die("could not read param file: density", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->accel));

  if (retval != 1) die("could not read param file: accel", __LINE__, __FILE__);

  retval = fscanf(fp, "%f\n", &(params->omega));

  if (retval != 1) die("could not read param file: omega", __LINE__, __FILE__);

  /* and close up the file */
  fclose(fp);

  /*
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

//  if(rank == MASTER){
    /* main grid */
    *cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

    if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed) * (params->ny * params->nx));

    if (*tmp_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr = malloc(sizeof(int) * (params->ny * params->nx));

    if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      /* centre */
      (*cells_ptr)[ii + jj*params->nx].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii + jj*params->nx].speeds[1] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[2] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[3] = w1;
      (*cells_ptr)[ii + jj*params->nx].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii + jj*params->nx].speeds[5] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[6] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[7] = w2;
      (*cells_ptr)[ii + jj*params->nx].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */
  for (int jj = 0; jj < params->ny; jj++)
  {
    for (int ii = 0; ii < params->nx; ii++)
    {
      (*obstacles_ptr)[ii + jj*params->nx] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile, "r");

  if (fp == NULL)
  {
    sprintf(message, "could not open input obstacles file: %s", obstaclefile);
    die(message, __LINE__, __FILE__);
  }

  /* read-in the blocked cells list */
  while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
  {
    /* some checks */
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);

    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);

    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    /* assign to array */
    (*obstacles_ptr)[xx + yy*params->nx] = blocked;
  }

  /* and close the file */
  fclose(fp);


  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float) * params->maxIters);

  return EXIT_SUCCESS;
}

int calc_nrows_from_rank(int rank, int size, int numRows)
{
  int nrows;

  int rest = numRows % size;
  nrows = numRows / size;       /* integer division */
  /* if there is a remainder */
  if (rank < rest)
    nrows += 1;  /* distrib remainding rows to each row starting from master */

  return nrows;
}


int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
             int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

  return final_av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  float total = 0.f;  /* accumulator */

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      for (int kk = 0; kk < NSPEEDS; kk++)
      {
        total += cells[ii + jj*params.nx].speeds[kk];
      }
    }
  }

  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int jj = 0; jj < params.ny; jj++)
  {
    for (int ii = 0; ii < params.nx; ii++)
    {
      /* an occupied cell */
      if (obstacles[ii + jj*params.nx])
      {
        u_x = u_y = u = 0.f;
        pressure = params.density * c_sq;
      }
      /* no obstacle */
      else
      {
        local_density = 0.f;

        for (int kk = 0; kk < NSPEEDS; kk++)
        {
          local_density += cells[ii + jj*params.nx].speeds[kk];
        }

        /* compute x velocity component */
        u_x = (cells[ii + jj*params.nx].speeds[1]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[8]
               - (cells[ii + jj*params.nx].speeds[3]
                  + cells[ii + jj*params.nx].speeds[6]
                  + cells[ii + jj*params.nx].speeds[7]))
              / local_density;
        /* compute y velocity component */
        u_y = (cells[ii + jj*params.nx].speeds[2]
               + cells[ii + jj*params.nx].speeds[5]
               + cells[ii + jj*params.nx].speeds[6]
               - (cells[ii + jj*params.nx].speeds[4]
                  + cells[ii + jj*params.nx].speeds[7]
                  + cells[ii + jj*params.nx].speeds[8]))
              / local_density;
        /* compute norm of velocity */
        u = sqrtf((u_x * u_x) + (u_y * u_y));
        /* compute pressure */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii * params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE, "w");

  if (fp == NULL)
  {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params.maxIters; ii++)
  {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
