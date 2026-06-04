#include "allvars.h"
#include "proto.h"
#include <vector>
#include <string>
#include <errno.h>
#include <string.h> // strerror


// ---------------------------
// helpers (file-local)
// ---------------------------
static inline double wrap_box(double x)
{
    // map to [0, LBOX)
    while (x >= (double)LBOX) x -= (double)LBOX;
    while (x <  0.0)          x += (double)LBOX;
    return x;
}

static void compute_group_means_periodic(const vector<int> &members,
                                         double Hpos[3], double Hvel[3])
{
    const double two_pi = 2.0 * M_PI;

    double sumc[3] = {0.0, 0.0, 0.0};   // cos sums for x,y,z
    double sums[3] = {0.0, 0.0, 0.0};   // sin sums for x,y,z
    double Vsum[3] = {0.0, 0.0, 0.0};   // velocity sums
    const int nmem = (int)members.size();

    for (int j = 0; j < nmem; ++j) {
        int idx = members[j];

        // positions -> angles on circle of circumference LBOX
        double x = Tracer[idx].Pos[0];
        double y = Tracer[idx].Pos[1];
        double z = Tracer[idx].Pos[2];

        double thx = two_pi * (x / (double)LBOX);
        double thy = two_pi * (y / (double)LBOX);
        double thz = two_pi * (z / (double)LBOX);

        sumc[0] += cos(thx);  sums[0] += sin(thx);
        sumc[1] += cos(thy);  sums[1] += sin(thy);
        sumc[2] += cos(thz);  sums[2] += sin(thz);

        // velocities: ordinary arithmetic mean
        Vsum[0] += Tracer[idx].Vel[0];
        Vsum[1] += Tracer[idx].Vel[1];
        Vsum[2] += Tracer[idx].Vel[2];
    }

    for (int k = 0; k < 3; ++k) {
        double ang = atan2(sums[k], sumc[k]);   // (-pi, pi]
        if (ang < 0.0) ang += two_pi;           // [0, 2pi)
        double coord = (ang / two_pi) * (double)LBOX;
        Hpos[k] = wrap_box(coord);
        Hvel[k] = Vsum[k] / (double)nmem;
    }
}

// ---------------------------
// I/O
// ---------------------------

void ReadTracers()
{
    FILE    *fd = NULL;
    clock_t t;

    fprintf(stdout, "\n Lectura de trazadores\n"); fflush(stdout);
    fprintf(stdout, " | File = %s\n", INPUTFILE.c_str()); fflush(stdout);

    t = clock();

    // ------------------------ stdin mode ----------------------------
    if (INPUTFILE == "-") {
        std::vector<tracers> tmp;
        tmp.reserve(1<<20); // optional pre-reserve

        fd = stdin;
        while (true) {
            tracers T;
            int n = fscanf(fd, "%d %lf %lf %lf %lf %lf %lf",
                           &T.ID,
                           &T.Pos[0], &T.Pos[1], &T.Pos[2],
                           &T.Vel[0], &T.Vel[1], &T.Vel[2]);
            if (n == EOF || n == 0) break;
            if (n != 7) {
                fprintf(stderr, "ERROR: malformed line while reading stdin\n");
                exit(EXIT_FAILURE);
            }
            T.GrpID = -1;
            T.pkey  = 0;
            tmp.push_back(T);
        }

        NTRAC = (int)tmp.size();
        NMEAN = (double)NTRAC / (LBOX*LBOX*LBOX);

        fprintf(stdout, " | Numero de trazadores    = %d \n", NTRAC);
        fprintf(stdout, " | Densidad media [h/Mpc]³ = %f \n", NMEAN);

        Tracer = (struct tracers*) malloc(NTRAC * sizeof(struct tracers));
        if (!Tracer) {
            fprintf(stderr, "ERROR: no memory for %d tracers\n", NTRAC);
            exit(EXIT_FAILURE);
        }
        for (int k = 0; k < NTRAC; k++) {
            Tracer[k] = tmp[k];
        }

        Time(t, "Read");
        return;
    }
    // ---------------------- end stdin mode ---------------------------


    // ---------------------- file mode (original) ---------------------
    NTRAC = CountLines(INPUTFILE.c_str());
    NMEAN = (double)NTRAC / (LBOX*LBOX*LBOX);

    fprintf(stdout, " | Numero de trazadores    = %d \n", NTRAC);
    fprintf(stdout, " | Densidad media [h/Mpc]³ = %f \n", NMEAN);

    Tracer = (struct tracers *) malloc(NTRAC * sizeof(struct tracers));
    if (!Tracer) {
        fprintf(stderr, "ERROR: no memory for %d tracers\n", NTRAC);
        exit(EXIT_FAILURE);
    }

    fd = fopen(INPUTFILE.c_str(), "r");
    if (!fd) {
        fprintf(stderr, "ERROR: cannot open input '%s': %s\n",
                INPUTFILE.c_str(), strerror(errno));
        exit(EXIT_FAILURE);
    }
    setvbuf(fd, NULL, _IOFBF, 1<<20);

    for (int i = 0; i < NTRAC; i++) {
        if (fscanf(fd, "%d %lf %lf %lf %lf %lf %lf",
                   &Tracer[i].ID,
                   &Tracer[i].Pos[0], &Tracer[i].Pos[1], &Tracer[i].Pos[2],
                   &Tracer[i].Vel[0], &Tracer[i].Vel[1], &Tracer[i].Vel[2]) != 7)
        {
            fprintf(stderr, "ERROR: malformed line %d in %s\n", i+1, INPUTFILE.c_str());
            fclose(fd);
            exit(EXIT_FAILURE);
        }
        Tracer[i].GrpID = -1;
        Tracer[i].pkey  = 0;
    }

    fclose(fd);
    Time(t, "Read");
}

// ======================================================================
// WriteGroups: unchanged except using GROUPFILE.c_str()
// ======================================================================
void WriteGroups()
{
    FILE    *fd;
    clock_t t;

    fprintf(stdout, "\n Escritura de grupos FoF\n"); fflush(stdout);
    fprintf(stdout, " | File = %s\n", GROUPFILE.c_str()); fflush(stdout);

    t = clock();

    fd = fopen(GROUPFILE.c_str(), "w");
    if (!fd) {
        fprintf(stderr, "ERROR: cannot open output '%s': %s\n",
                GROUPFILE.c_str(), strerror(errno));
        exit(EXIT_FAILURE);
    }
    setvbuf(fd, NULL, _IOFBF, 1<<20);

    const int ngr = (int)G.size();
    fprintf(stdout, " | Total de grupos encontrados = %d \n", ngr); fflush(stdout);

    fprintf(fd, "Particle_ID Px Py Pz PVx PVy PVz Halo_ID Hx Hy Hz HVx HVy HVz \n");

    int gr_kept = 0;
    for (int i = 0; i < ngr; i++) {
        int nmem = (int)G[i].Mem.size();
        if (nmem < MINGROUP) continue;

        double Hpos[3], Hvel[3];
        compute_group_means_periodic(G[i].Mem, Hpos, Hvel);

        for (int j = 0; j < nmem; j++) {
            int idx = G[i].Mem[j];
            fprintf(fd, "%d %lf %lf %lf ",
                    Tracer[idx].ID,
                    Tracer[idx].Pos[0], Tracer[idx].Pos[1], Tracer[idx].Pos[2]);
            fprintf(fd, "%lf %lf %lf ",
                    Tracer[idx].Vel[0], Tracer[idx].Vel[1], Tracer[idx].Vel[2]);
            fprintf(fd, "%d %lf %lf %lf ",
                    i, Hpos[0], Hpos[1], Hpos[2]);
            fprintf(fd, "%lf %lf %lf \n",
                    Hvel[0], Hvel[1], Hvel[2]);
        }
        gr_kept++;
    }

    fclose(fd);

    fprintf(stdout, " | Total de grupos con mas de %d miembros = %d \n\n",
            MINGROUP, gr_kept); fflush(stdout);

    Time(t, "Write");
}
