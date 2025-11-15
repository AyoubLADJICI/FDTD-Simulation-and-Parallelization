#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>

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
              double dy, double val)
{
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

int write_data_vtk(struct data *data, int step, int rank)
{
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
                       int numranks)
{
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

int main(int argc, char **argv)
{
  if(argc != 2) {
    printf("Usage: %s problem_id\n", argv[0]);
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

  printf("Solving problem %d:\n", problem_id);
  printf(" - space %gm x %gm (dx=%g, dy=%g; nx=%d, ny=%d)\n",
         dx * nx, dy * ny, dx, dy, nx, ny);
  printf(" - time %gs (dt=%g, nt=%d)\n", dt * nt, dt, nt);

  struct data ez, hx, hy;
  if(init_data(&ez, "ez", nx, ny, dx, dy, 0.) ||
     init_data(&hx, "hx", nx, ny - 1, dx, dy, 0.) ||
     init_data(&hy, "hy", nx - 1, ny, dx, dy, 0.)) {
    printf("Error: could not allocate data\n");
    return 1;
  }

  double start = GET_TIME();

  for(int n = 0; n < nt; n++) {
    if(n && (n % (nt / 10)) == 0) {
      double time_sofar = GET_TIME() - start;
      double eta = (nt - n) * time_sofar / n;
      printf("Computing time step %d/%d (ETA: %g seconds)     \r", n, nt, eta);
      fflush(stdout);
    }

    // update hx and hy
    double chy = dt / (dy * mu);
    for(int i = 0; i < nx; i++) {
      for(int j = 0; j < ny - 1; j++) {
        double hx_ij =
          GET(&hx, i, j) - chy * (GET(&ez, i, j + 1) - GET(&ez, i, j));
        SET(&hx, i, j, hx_ij);
      }
    }
    double chx = dt / (dx * mu);
    for(int i = 0; i < nx - 1; i++) {
      for(int j = 0; j < ny; j++) {
        double hy_ij =
          GET(&hy, i, j) + chx * (GET(&ez, i + 1, j) - GET(&ez, i, j));
        SET(&hy, i, j, hy_ij);
      }
    }

    // update ez
    double cex = dt / (dx * eps), cey = dt / (dy * eps);
    for(int i = 1; i < nx - 1; i++) {
      for(int j = 1; j < ny - 1; j++) {
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
      // sinusoidal excitation at 2.4 GHz in the middle of the domain
      SET(&ez, nx / 2, ny / 2, sin(2. * M_PI * 2.4e9 * t));
      break;
    default:
      printf("Error: unknown source\n");
      break;
    }

    // output step data in VTK format
    if(sampling_rate && !(n % sampling_rate)) {
      write_data_vtk(&ez, n, 0);
      // write_data_vtk(&hx, n, 0);
      // write_data_vtk(&hy, n, 0);
    }
  }

  // write VTK manifest, linking to individual step data files
  write_manifest_vtk("ez", dt, nt, sampling_rate, 1);
  // write_manifest_vtk("hx", dt, nt, sampling_rate, 1);
  // write_manifest_vtk("hy", dt, nt, sampling_rate, 1);

  double time = GET_TIME() - start;
  printf("\nDone: %g seconds (%g MUpdates/s)\n", time,
         1.e-6 * (double)nx * (double)ny * (double)nt / time);

  free_data(&ez);
  free_data(&hx);
  free_data(&hy);

  return 0;
}

