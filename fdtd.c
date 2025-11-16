#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <limits.h>
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

void exchange_halo_ez(MPI_Comm comm_cart, int rank_x, int rank_y, int Px, int Py,
                      struct data *ez, int nx_local, int ny_local) {
    // Obtenir les rangs des voisins
    int rank_north, rank_south, rank_east, rank_west;
    MPI_Cart_shift(comm_cart, 0, 1, &rank_west, &rank_east);  // dim 0 pour X
    MPI_Cart_shift(comm_cart, 1, 1, &rank_south, &rank_north); // dim 1 pour Y

    // Tampons de communication pour EZ (dimensions: nx_local pour vertical, ny_local pour horizontal)
    // On a besoin de buffers pour 4 directions (send et recv)
    double *send_buffer_west = (double*) malloc(ny_local * sizeof(double));
    double *recv_buffer_west = (double*) malloc(ny_local * sizeof(double));
    double *send_buffer_east = (double*) malloc(ny_local * sizeof(double));
    double *recv_buffer_east = (double*) malloc(ny_local * sizeof(double));

    double *send_buffer_south = (double*) malloc(nx_local * sizeof(double));
    double *recv_buffer_south = (double*) malloc(nx_local * sizeof(double));
    double *send_buffer_north = (double*) malloc(nx_local * sizeof(double));
    double *recv_buffer_north = (double*) malloc(nx_local * sizeof(double));

    // PACKING : Copier les données des bords internes de EZ vers les tampons d'envoi
    // BORD OUEST (colonne 1 interne)
    for (int j = 0; j < ny_local; j++) send_buffer_west[j] = GET(ez, 1, j + 1);
    // BORD EST (colonne nx_local interne)
    for (int j = 0; j < ny_local; j++) send_buffer_east[j] = GET(ez, nx_local, j + 1);
    // BORD SUD (rangée 1 interne)
    for (int i = 0; i < nx_local; i++) send_buffer_south[i] = GET(ez, i + 1, 1);
    // BORD NORD (rangée ny_local interne)
    for (int i = 0; i < nx_local; i++) send_buffer_north[i] = GET(ez, i + 1, ny_local);

    // COMMUNICATIONS (envoi/réception)
    // Utiliser des paires de tags différentes pour éviter les deadlocks et distinguer les messages
    // Par exemple, 0/1 pour X, 2/3 pour Y.
    // Échange WEST-EAST (colonnes)
    if (rank_west != MPI_PROC_NULL) {
        MPI_Send(send_buffer_west, ny_local, MPI_DOUBLE, rank_west, 0, comm_cart);
        MPI_Recv(recv_buffer_west, ny_local, MPI_DOUBLE, rank_west, 1, comm_cart, MPI_STATUS_IGNORE);
    }
    if (rank_east != MPI_PROC_NULL) {
        MPI_Send(send_buffer_east, ny_local, MPI_DOUBLE, rank_east, 1, comm_cart);
        MPI_Recv(recv_buffer_east, ny_local, MPI_DOUBLE, rank_east, 0, comm_cart, MPI_STATUS_IGNORE);
    }
    // Échange SOUTH-NORTH (rangées)
    if (rank_south != MPI_PROC_NULL) {
        MPI_Send(send_buffer_south, nx_local, MPI_DOUBLE, rank_south, 2, comm_cart);
        MPI_Recv(recv_buffer_south, nx_local, MPI_DOUBLE, rank_south, 3, comm_cart, MPI_STATUS_IGNORE);
    }
    if (rank_north != MPI_PROC_NULL) {
        MPI_Send(send_buffer_north, nx_local, MPI_DOUBLE, rank_north, 3, comm_cart);
        MPI_Recv(recv_buffer_north, nx_local, MPI_DOUBLE, rank_north, 2, comm_cart, MPI_STATUS_IGNORE);
    }

    // UNPACKING : Copier les données reçues dans les halos de EZ
    // HALO OUEST (colonne 0)
    if (rank_west != MPI_PROC_NULL) {
        for (int j = 0; j < ny_local; j++) SET(ez, 0, j + 1, recv_buffer_west[j]);
    }
    // HALO EST (colonne nx_local + 1)
    if (rank_east != MPI_PROC_NULL) {
        for (int j = 0; j < ny_local; j++) SET(ez, nx_local + 1, j + 1, recv_buffer_east[j]);
    }
    // HALO SUD (rangée 0)
    if (rank_south != MPI_PROC_NULL) {
        for (int i = 0; i < nx_local; i++) SET(ez, i + 1, 0, recv_buffer_south[i]);
    }
    // HALO NORD (rangée ny_local + 1)
    if (rank_north != MPI_PROC_NULL) {
        for (int i = 0; i < nx_local; i++) SET(ez, i + 1, ny_local + 1, recv_buffer_north[i]);
    }

    free(send_buffer_west); free(recv_buffer_west);
    free(send_buffer_east); free(recv_buffer_east);
    free(send_buffer_south); free(recv_buffer_south);
    free(send_buffer_north); free(recv_buffer_north);
}

void exchange_halo_h(MPI_Comm comm_cart, int rank_x, int rank_y, int Px, int Py,
                     struct data *hx, struct data *hy, int nx_local, int ny_local, 
                     int real_local_ny_hx, int real_local_nx_hy, 
                     int real_local_nx_ez, int real_local_ny_ez) {
    // Obtenir les rangs des voisins (mêmes que pour EZ)
    int rank_north, rank_south, rank_east, rank_west;
    MPI_Cart_shift(comm_cart, 0, 1, &rank_west, &rank_east);
    MPI_Cart_shift(comm_cart, 1, 1, &rank_south, &rank_north);

    // --- Tampons et échanges pour Hx ---
    // Hx n'a pas de halo en Y car sa taille "réelle" est ny_local+1 (de 0 à ny_local)
    // Hx a un halo en X (colonnes 0 et nx_local+1)
    // Send/Recv pour Hx (colonnes)
    double *send_hx_west = (double*) malloc(real_local_ny_hx * sizeof(double));
    double *recv_hx_west = (double*) malloc(real_local_ny_hx * sizeof(double));
    double *send_hx_east = (double*) malloc(real_local_ny_hx * sizeof(double));
    double *recv_hx_east = (double*) malloc(real_local_ny_hx * sizeof(double));

    // PACKING Hx : Colonnes 1 (ouest) et nx_local (est)
    for (int j = 0; j < real_local_ny_hx; j++) send_hx_west[j] = GET(hx, 1, j);
    for (int j = 0; j < real_local_ny_hx; j++) send_hx_east[j] = GET(hx, nx_local, j);

    // COMMUNICATIONS Hx
    if (rank_west != MPI_PROC_NULL) {
        MPI_Send(send_hx_west, real_local_ny_hx, MPI_DOUBLE, rank_west, 4, comm_cart); // tag 4
        MPI_Recv(recv_hx_west, real_local_ny_hx, MPI_DOUBLE, rank_west, 5, comm_cart, MPI_STATUS_IGNORE); // tag 5
    }
    if (rank_east != MPI_PROC_NULL) {
        MPI_Send(send_hx_east, real_local_ny_hx, MPI_DOUBLE, rank_east, 5, comm_cart); // tag 5
        MPI_Recv(recv_hx_east, real_local_ny_hx, MPI_DOUBLE, rank_east, 4, comm_cart, MPI_STATUS_IGNORE); // tag 4
    }

    // UNPACKING Hx : Halos Hx (colonnes 0 et nx_local+1)
    if (rank_west != MPI_PROC_NULL) {
        for (int j = 0; j < real_local_ny_hx; j++) SET(hx, 0, j, recv_hx_west[j]);
    }
    if (rank_east != MPI_PROC_NULL) {
        for (int j = 0; j < real_local_ny_hx; j++) SET(hx, nx_local + 1, j, recv_hx_east[j]);
    }
    free(send_hx_west); free(recv_hx_west); free(send_hx_east); free(recv_hx_east);


    // --- Tampons et échanges pour Hy ---
    // Hy n'a pas de halo en X car sa taille "réelle" est nx_local+1 (de 0 à nx_local)
    // Hy a un halo en Y (rangées 0 et ny_local+1)
    // Send/Recv pour Hy (rangées)
    double *send_hy_south = (double*) malloc(real_local_nx_hy * sizeof(double));
    double *recv_hy_south = (double*) malloc(real_local_nx_hy * sizeof(double));
    double *send_hy_north = (double*) malloc(real_local_nx_hy * sizeof(double));
    double *recv_hy_north = (double*) malloc(real_local_nx_hy * sizeof(double));

    // PACKING Hy : Rangées 1 (sud) et ny_local (nord)
    for (int i = 0; i < real_local_nx_hy; i++) send_hy_south[i] = GET(hy, i, 1);
    for (int i = 0; i < real_local_nx_hy; i++) send_hy_north[i] = GET(hy, i, ny_local);

    // COMMUNICATIONS Hy
    if (rank_south != MPI_PROC_NULL) {
        MPI_Send(send_hy_south, real_local_nx_hy, MPI_DOUBLE, rank_south, 6, comm_cart); // tag 6
        MPI_Recv(recv_hy_south, real_local_nx_hy, MPI_DOUBLE, rank_south, 7, comm_cart, MPI_STATUS_IGNORE); // tag 7
    }
    if (rank_north != MPI_PROC_NULL) {
        MPI_Send(send_hy_north, real_local_nx_hy, MPI_DOUBLE, rank_north, 7, comm_cart); // tag 7
        MPI_Recv(recv_hy_north, real_local_nx_hy, MPI_DOUBLE, rank_north, 6, comm_cart, MPI_STATUS_IGNORE); // tag 6
    }

    // UNPACKING Hy : Halos Hy (rangées 0 et ny_local+1)
    if (rank_south != MPI_PROC_NULL) {
        for (int i = 0; i < real_local_nx_hy; i++) SET(hy, i, 0, recv_hy_south[i]);
    }
    if (rank_north != MPI_PROC_NULL) {
        for (int i = 0; i < real_local_nx_hy; i++) SET(hy, i, ny_local + 1, recv_hy_north[i]);
    }
    free(send_hy_south); free(recv_hy_south); free(send_hy_north); free(recv_hy_north);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int num_ranks, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks); // Get number of ranks
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get current rank

  if (rank == 0) printf("Number of MPI ranks: %d\n", num_ranks);

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
  const char* s_env = getenv("CFL");
  double S = s_env ? atof(s_env) : 0.90;
  if (rank == 0) {
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Rank 0 running on processor %s\n", processor_name);
    printf("Using stability factor S = %g\n", S);
  }

  switch(problem_id) {
  case 1: // small size problem
    dx = dy = (3.e8 / 2.4e9) / 20.; // wavelength / 20
    nx = ny = 500;
    dt = S / (3.e8 * sqrt(1. / (dx * dx) + 1. / (dy * dy))); // cfl / 2
    nt = 500;
    sampling_rate = 5; // save 1 step out of 5
    break;
  case 2: // larger size problem, usable for initial scaling tests
    dx = dy = (3.e8 / 2.4e9) / 40.; // wavelength / 40
    nx = ny = 16000;
    dt = S / (3.e8 * sqrt(1. / (dx * dx) + 1. / (dy * dy))); // cfl / 2
    nt = 500;
    sampling_rate = 0; // don't save results
    break;
  default:
    if (rank == 0) {
      printf("Error: unknown problem id %d\n", problem_id);
    }
    MPI_Finalize();
    return 1;
  }

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

  MPI_Comm comm_cart;
  int dims[2] = {Px, Py};
  int periods[2] = {0, 0}; // non-periodic boundaries
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &comm_cart);

  int coords[2];
  MPI_Cart_coords(comm_cart, rank, 2, coords);
  int rank_x = coords[0];
  int rank_y = coords[1];

  int rank_north, rank_south, rank_east, rank_west;
  MPI_Cart_shift(comm_cart, 0, 1, &rank_west, &rank_east);  // Direction X (dim 0), shift de 1
  MPI_Cart_shift(comm_cart, 1, 1, &rank_south, &rank_north); // Direction Y (dim 1), shift de 1

  int nx_local, ny_local;
  int offset_x, offset_y;

  nx_local = nx / Px;
  ny_local = ny / Py;

  offset_x = rank_x * nx_local;
  offset_y = rank_y * ny_local;

  printf("Rank %d coordinates (%d,%d) offset (%d,%d) local size (%d,%d)\n",
         rank, rank_x, rank_y, offset_x, offset_y, nx_local, ny_local);

  int ez_alloc_nx = nx_local;
  int ez_alloc_ny = ny_local;
  int hx_alloc_nx = nx_local;
  int hx_alloc_ny = ny_local - 1;
  int hy_alloc_nx = nx_local - 1;
  int hy_alloc_ny = ny_local;

  int local_ez_nx_with_halo = nx_local + (rank_x == 0 ? 1 : 0) + (rank_x == Px - 1 ? 1 : 0);
  int local_ez_ny_with_halo = ny_local + (rank_y == 0 ? 1 : 0) + (rank_y == Py - 1 ? 1 : 0);

  int real_local_nx_ez = nx_local + 2;
  int real_local_ny_ez = ny_local + 2;

  int real_local_nx_hx = real_local_nx_ez;
  int real_local_ny_hx = real_local_ny_ez - 1;

  int real_local_nx_hy = real_local_nx_ez - 1;
  int real_local_ny_hy = real_local_ny_ez;

  struct data ez, hx, hy;
  if(init_data(&ez, "ez", real_local_nx_ez, real_local_ny_ez, dx, dy, 0.) ||
     init_data(&hx, "hx", real_local_nx_hx, real_local_ny_hx, dx, dy, 0.) ||
     init_data(&hy, "hy", real_local_nx_hy, real_local_ny_hy, dx, dy, 0.)) {
    printf("Error: could not allocate data on rank %d\n", rank);
    MPI_Finalize();
    return 1;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  double start = MPI_Wtime();

  for(int n = 0; n < nt; n++) {
    if(n && (n % (nt / 10)) == 0) {
      double time_sofar = MPI_Wtime() - start;
      double eta = (nt - n) * time_sofar / n;
      if (rank == 0) {
        printf("Computing time step %d/%d (ETA: %g seconds) \r", n, nt, eta);
        fflush(stdout);
      }
    }

    exchange_halo_ez(comm_cart, rank_x, rank_y, Px, Py, &ez, nx_local, ny_local);
    
    // update hx and hy
    double chy = dt / (dy * mu);
    for(int i = 1; i < nx_local + 1; i++) {
      for(int j = 1; j < ny_local; j++) {
        double hx_ij =
          GET(&hx, i, j) - chy * (GET(&ez, i, j + 1) - GET(&ez, i, j));
        SET(&hx, i, j, hx_ij);
      }
    }
    
    double chx = dt / (dx * mu);
    for(int i = 1; i < nx_local; i++) {
      for(int j = 1; j < ny_local + 1; j++) {
        double hy_ij =
          GET(&hy, i, j) + chx * (GET(&ez, i + 1, j) - GET(&ez, i, j));
        SET(&hy, i, j, hy_ij);
      }
    }
    
    exchange_halo_h(comm_cart, rank_x, rank_y, Px, Py, &hx, &hy, nx_local, ny_local, 
      real_local_nx_hx, real_local_ny_hx, real_local_nx_ez, real_local_ny_ez);

    // update ez
    double cex = dt / (dx * eps), cey = dt / (dy * eps);
    for(int i = 1; i < nx_local + 1; i++) {
      for(int j = 1; j < ny_local + 1; j++) {
        double ez_ij = GET(&ez, i, j) +
                       cex * (GET(&hy, i, j) - GET(&hy, i - 1, j)) -
                       cey * (GET(&hx, i, j) - GET(&hx, i, j - 1));
        SET(&ez, i, j, ez_ij);
      }
    }
  
    // impose source
    double t = n * dt;
    switch(problem_id) {
    case 1:
    case 2:
      int source_x_global = nx / 2;
      int source_y_global = ny / 2;

      if (source_x_global >= offset_x && source_x_global < offset_x + nx_local &&
          source_y_global >= offset_y && source_y_global < offset_y + ny_local) {
        int source_x_local = source_x_global - offset_x; 
        int source_y_local = source_y_global - offset_y; 
        SET(&ez, source_x_local + 1, source_y_local + 1, sin(2. * M_PI * 2.4e9 * t));
      }
      break;
    default:
      if (rank == 0) printf("Error: unknown source\n");
      break;
    }

    // Si le rang est sur le bord gauche global (rank_x == 0)
    if (rank_x == 0) {
        for (int j = 0; j < real_local_ny_ez; j++) { // Tous les y
            SET(&ez, 1, j, 0.); // La première colonne interne (à droite du halo 0)
        }
    }
    // Si le rang est sur le bord droit global (rank_x == Px - 1)
    if (rank_x == Px - 1) {
        for (int j = 0; j < real_local_ny_ez; j++) {
            SET(&ez, nx_local, j, 0.); // La dernière colonne interne (à gauche du halo nx_local+1)
        }
    }
    // Si le rang est sur le bord bas global (rank_y == 0)
    if (rank_y == 0) {
        for (int i = 0; i < real_local_nx_ez; i++) {
            SET(&ez, i, 1, 0.); // La première rangée interne
        }
    }
    // Si le rang est sur le bord haut global (rank_y == Py - 1)
    if (rank_y == Py - 1) {
        for (int i = 0; i < real_local_nx_ez; i++) {
            SET(&ez, i, ny_local, 0.); // La dernière rangée interne
        }
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

