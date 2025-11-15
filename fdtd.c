#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

#if defined(_OPENMP)
#include <omp.h>
#define GET_TIME() (omp_get_wtime()) // wall time
#else
#define GET_TIME() ((double)clock() / CLOCKS_PER_SEC) // cpu time
#endif

struct data {
  const char *name;
  int nx, ny;
  double dx, dy, *values;
};

#define GET(data, i, j) ((data)->values[(data)->nx * (j) + (i)])
#define SET(data, i, j, val) ((data)->values[(data)->nx * (j) + (i)] = (val))

int init_data(struct data *data, const char *name, int nx, int ny, double dx,
              double dy, double val) {
  data->name = name;
  data->nx = nx;
  data->ny = ny;
  data->dx = dx;
  data->dy = dy;
  data->values = (double *)malloc(nx * ny * sizeof(double));
  if(!data->values) {
    printf("Error: Could not allocate data\n");
    return 1;
  }
  for(int i = 0; i < nx * ny; i++) data->values[i] = val;
  return 0;
}

void free_data(struct data *data) { free(data->values); }

int write_data_vtk(struct data *data, int step, int rank) {
  char out[512];
  if(strlen(data->name) > 256) {
    printf("Error: data name too long for output VTK file\n");
    return 1;
  }
  sprintf(out, "%s_rank%d_%d.vti", data->name, rank, step);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK file '%s'\n", out);
    return 1;
  }

  uint64_t num_points = data->nx * data->ny;
  uint64_t num_bytes = num_points * sizeof(double);

  fprintf(fp, "<?xml version=\"1.0\"?>\n"
              "<VTKFile"
              " type=\"ImageData\""
              " version=\"1.0\""
              " byte_order=\"LittleEndian\""
              " header_type=\"UInt64\""
              ">\n"
              "  <ImageData"
              " WholeExtent=\"0 %d 0 %d 0 %d\""
              " Spacing=\"%lf %lf %lf\""
              " Origin=\"%lf %lf %lf\""
              ">\n"
              "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
              "      <PointData Scalars=\"scalar_data\">\n"
              "        <DataArray"
              " type=\"Float64\""
              " Name=\"%s\""
              " format=\"appended\""
              " offset=\"0\""
              ">\n"
              "        </DataArray>\n"
              "      </PointData>\n"
              "    </Piece>\n"
              "  </ImageData>\n"
              "  <AppendedData encoding=\"raw\">\n_",
          data->nx - 1, data->ny - 1, 0,
          data->dx, data->dy, 0.,
          0., 0., 0.,
          data->nx - 1, data->ny - 1, 0,
          data->name);

  fwrite(&num_bytes, sizeof(uint64_t), 1, fp);
  fwrite(data->values, sizeof(double), num_points, fp);

  fprintf(fp, "  </AppendedData>\n"
              "</VTKFile>\n");

  fclose(fp);

  return 0;
}

int write_manifest_vtk(const char *name, double dt, int nt, int sampling_rate,
                       int numranks) {
  char out[512];
  if(strlen(name) > 256) {
    printf("Error: name too long for Paraview manifest file\n");
    return 1;
  }
  sprintf(out, "%s.pvd", name);

  FILE *fp = fopen(out, "wb");
  if(!fp) {
    printf("Error: Could not open output VTK manifest file '%s'\n", out);
    return 1;
  }

  fprintf(fp, "<VTKFile"
              " type=\"Collection\""
              " version=\"0.1\""
              " byte_order=\"LittleEndian\">\n"
              "  <Collection>\n");

  for(int n = 0; n < nt; n++) {
    if(sampling_rate && !(n % sampling_rate)) {
      double t = n * dt;
      for(int rank = 0; rank < numranks; rank++) {
        fprintf(fp, "    <DataSet"
                    " timestep=\"%g\""
                    " part=\"%d\""
                    " file='%s_rank%d_%d.vti'/>\n",
                t, rank, name, rank, n);
      }
    }
  }

  fprintf(fp, "  </Collection>\n"
              "</VTKFile>\n");
  fclose(fp);
  return 0;
}

void find_best_partition(int nx, int ny, int num_ranks, int* Px_out, int* Py_out) {
    double best_score = (double)INT32_MAX;
    int best_Px = -1;
    int best_Py = -1;

    for (int Px = 1; Px <= num_ranks; Px++) {

        if (num_ranks % Px != 0)
            continue;

        int Py = num_ranks / Px;

        double sub_x = (double)nx / Px;
        double sub_y = (double)ny / Py;

        double ratio = sub_x / sub_y;
        double score = fabs(1.0 - ratio);
        
        if (score < best_score) {
            best_score = score;
            best_Px = Px;
            best_Py = Py;
        }
    }

    *Px_out = best_Px;
    *Py_out = best_Py;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int num_ranks;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks); // Get number of ranks
  printf("Number of MPI ranks: %d\n", num_ranks);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current rank
  printf("Current MPI rank: %d\n", rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len); // Get processor name
  printf("Processor name: %s\n", processor_name);

  if(argc != 2) {
    if (rank == 0) {
      printf("Usage: %s problem_id\n", argv[0]);
    }
    MPI_Finalize();
    return 1;
  }

  double dx = 1., dy = 1., dt = 1.;
  int nx = 1, ny = 1, nt = 1, sampling_rate = 1;

  double eps = 8.854187817e-12;
  double mu = 1.2566370614359173e-06;

  int problem_id = atoi(argv[1]);
  switch(problem_id) {
  case 1: // small size problem
    dx = dy = (3.e8 / 2.4e9) / 20.; // wavelength / 20
    nx = ny = 500;
    dt = 0.5 / (3.e8 * sqrt(1. / (dx * dx) + 1. / (dy * dy))); // cfl / 2
    nt = 500;
    sampling_rate = 5; // save 1 step out of 5
    break;
  case 2: // larger size problem, usable for initial scaling tests
    dx = dy = (3.e8 / 2.4e9) / 40.; // wavelength / 40
    nx = ny = 16000;
    dt = 0.5 / (3.e8 * sqrt(1. / (dx * dx) + 1. / (dy * dy))); // cfl / 2
    nt = 500;
    sampling_rate = 0; // don't save results
    break;
  default:
    printf("Error: unknown problem id %d\n", problem_id);
    return 1;
  }

  // We want that only rank 0 prints the initial information
  if (rank == 0) {
    printf("Solving problem %d:\n", problem_id);
    printf(" - space %gm x %gm (dx=%g, dy=%g; nx=%d, ny=%d)\n",
          dx * nx, dy * ny, dx, dy, nx, ny);
    printf(" - time %gs (dt=%g, nt=%d)\n", dt * nt, dt, nt);
  }

  int Px, Py;
  find_best_partition(nx, ny, num_ranks, &Px, &Py);
  if (rank == 0) {
    printf("Using a %d x %d partitioning of the domain\n", Px, Py);
  }

  // determine this rank's position in the Px x Py grid
  int rank_x = rank % Px;
  int rank_y = rank / Px;

  // determine neighboring ranks
  const int left_rank = rank_x > 0 ? rank - 1 : MPI_PROC_NULL;
  const int right_rank = rank_x < Px - 1 ? rank + 1 : MPI_PROC_NULL;
  const int down_rank = rank_y < Py - 1 ? rank + Px : MPI_PROC_NULL;
  const int up_rank = rank_y > 0 ? rank - Px : MPI_PROC_NULL;

  // determine this rank's local domain size
  int local_nx = (nx / Px) + 2; // +2 for ghost cells
  int local_ny = (ny / Py) + 2;
  
  if (rank_x == Px - 1) local_nx = (nx - (nx / Px) * (Px - 1)) + 2;
  if (rank_y == Py - 1) local_ny = (ny - (ny / Py) * (Py - 1)) + 2;

  int global_i_start = rank_x * (nx / Px);
  int global_j_start = rank_y * (ny / Py);

  struct data ez, hx, hy;
  if(init_data(&ez, "ez", local_nx, local_ny, dx, dy, 0.) ||
     init_data(&hx, "hx", local_nx, local_ny, dx, dy, 0.) ||
     init_data(&hy, "hy", local_nx, local_ny, dx, dy, 0.)) {
    printf("Error: could not allocate data on rank %d\n", rank);
    MPI_Finalize();
    return 1;
  }
  
  MPI_Datatype hy_col_type;
  MPI_Type_vector(local_ny - 2, 1, local_nx, MPI_DOUBLE, &hy_col_type);
  MPI_Type_commit(&hy_col_type);
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  MPI_Request send_requests[4];
  MPI_Request recv_requests[4];
  MPI_Status  statuses[4]; 

  for(int n = 0; n < nt; n++) {
    if(rank == 0 && n && (n % (nt / 10)) == 0) {
      double time_sofar = MPI_Wtime() - start;
      double eta = (nt - n) * time_sofar / n;
      printf("Computing time step %d/%d (ETA: %g seconds) \r", n, nt, eta);
      fflush(stdout);
    }

    // update hx and hy
    double chy = dt / (dy * mu);
    for(int i = 1; i < local_nx - 1; i++) {
      for(int j = 1; j < local_ny - 2; j++) {
        double hx_ij =
          GET(&hx, i, j) - chy * (GET(&ez, i, j + 1) - GET(&ez, i, j));
        SET(&hx, i, j, hx_ij);
      }
    }
    
    double chx = dt / (dx * mu);
    for(int i = 1; i < local_nx - 2; i++) {
      for(int j = 1; j < local_ny - 1; j++) {
        double hy_ij =
          GET(&hy, i, j) + chx * (GET(&ez, i + 1, j) - GET(&ez, i, j));
        SET(&hy, i, j, hy_ij);
      }
    }

    // Recevoir les fantômes
    MPI_Irecv(&GET(&hx, 1, 0), local_nx - 2, MPI_DOUBLE, down_rank, 0, MPI_COMM_WORLD, &recv_requests[0]);
    MPI_Irecv(&GET(&hx, 1, local_ny - 1), local_nx - 2, MPI_DOUBLE, up_rank, 1, MPI_COMM_WORLD, &recv_requests[1]); 
    MPI_Irecv(&GET(&hy, 0, 1), 1, hy_col_type, left_rank, 2, MPI_COMM_WORLD, &recv_requests[2]); 
    MPI_Irecv(&GET(&hy, local_nx - 1, 1), 1, hy_col_type, right_rank, 3, MPI_COMM_WORLD, &recv_requests[3]); 

    // Envoyer nos bordures
    MPI_Isend(&GET(&hx, 1, 1), local_nx - 2, MPI_DOUBLE, down_rank, 1, MPI_COMM_WORLD, &send_requests[0]); 
    MPI_Isend(&GET(&hx, 1, local_ny - 2), local_nx - 2, MPI_DOUBLE, up_rank, 0, MPI_COMM_WORLD, &send_requests[1]); 
    MPI_Isend(&GET(&hy, 1, 1), 1, hy_col_type, left_rank,  3, MPI_COMM_WORLD, &send_requests[2]); 
    MPI_Isend(&GET(&hy, local_nx - 2, 1), 1, hy_col_type, right_rank, 2, MPI_COMM_WORLD, &send_requests[3]); 

    // update ez
    double cex = dt / (dx * eps), cey = dt / (dy * eps);
    for(int i = 2; i < local_nx - 2; i++) {
      for(int j = 2; j < local_ny - 2; j++) {
        double ez_ij = GET(&ez, i, j) +
                       cex * (GET(&hy, i, j) - GET(&hy, i - 1, j)) -
                       cey * (GET(&hx, i, j) - GET(&hx, i, j - 1));
        SET(&ez, i, j, ez_ij);
      }
    }

    MPI_Waitall(4, recv_requests, statuses);

    for(int j = 1; j < local_ny - 1; j++) {
      double ez_ij = GET(&ez, 1, j) +
                     cex * (GET(&hy, 1, j) - GET(&hy, 0, j)) - // hy[0,j] is ghost
                     cey * (GET(&hx, 1, j) - GET(&hx, 1, j - 1));
      SET(&ez, 1, j, ez_ij);
    }

    for(int j = 1; j < local_ny - 1; j++) {
      double ez_ij = GET(&ez, local_nx - 2, j) +
                     cex * (GET(&hy, local_nx - 2, j) - GET(&hy, local_nx - 3, j)) -
                     cey * (GET(&hx, local_nx - 2, j) - GET(&hx, local_nx - 2, j - 1));
      SET(&ez, local_nx - 2, j, ez_ij);
    }

    for(int i = 2; i < local_nx - 2; i++) { 
      double ez_ij = GET(&ez, i, 1) +
                     cex * (GET(&hy, i, 1) - GET(&hy, i - 1, 1)) -
                     cey * (GET(&hx, i, 1) - GET(&hx, i, 0)); // hx[i,0] is ghost
      SET(&ez, i, 1, ez_ij);
    }

    for(int i = 2; i < local_nx - 2; i++) { 
      double ez_ij = GET(&ez, i, local_ny - 2) +
                     cex * (GET(&hy, i, local_ny - 2) - GET(&hy, i - 1, local_ny - 2)) -
                     cey * (GET(&hx, i, local_ny - 2) - GET(&hx, i, local_ny - 3));
      SET(&ez, i, local_ny - 2, ez_ij);
    }

    // impose source
    double t = n * dt;
    switch(problem_id) {
    case 1:
    case 2:
      int source_i_global = nx / 2;
      int source_j_global = ny / 2;

      if (source_i_global >= global_i_start &&
          source_i_global < (global_i_start + local_nx - 2) &&
          source_j_global >= global_j_start &&
          source_j_global < (global_j_start + local_ny - 2)) {
        int source_i_local = (source_i_global - global_i_start) + 1; // +1 for ghost cell
        int source_j_local = (source_j_global - global_j_start) + 1; // +1 for ghost cell
      // sinusoidal excitation at 2.4 GHz in the middle of the domain
      SET(&ez, source_i_local, source_j_local, sin(2. * M_PI * 2.4e9 * t));
      }
      //SET(&ez, nx / 2, ny / 2, sin(2. * M_PI * 2.4e9 * t));
      break;
    default:
      if (rank == 0) printf("Error: unknown source\n");
      break;
    }

    // output step data in VTK format
    if(sampling_rate && !(n % sampling_rate)) {
      write_data_vtk(&ez, n, rank);
      // write_data_vtk(&ez, n, 0);
      // write_data_vtk(&hx, n, 0);
      // write_data_vtk(&hy, n, 0);
    }
  }

  // write VTK manifest, linking to individual step data files
  if (rank == 0) {
    write_manifest_vtk("ez", dt, nt, sampling_rate, num_ranks);
    // write_manifest_vtk("hx", dt, nt, sampling_rate, 1);
    // write_manifest_vtk("hy", dt, nt, sampling_rate, 1);
  }

  double time = MPI_Wtime() - start;
  if (rank == 0) {
    printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
           1.e-6 * (double)nx * (double)ny * (double)nt / time);
  }

  free_data(&ez);
  free_data(&hx);
  free_data(&hy);

  MPI_Finalize();
  return 0;
}

