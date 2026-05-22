#include <omp.h>
#include "allvars.h"
#include "proto.h"

// ------------------------------------------------------------
// Voxel-level FoF (OpenMP BFS on voxel graph)
// ------------------------------------------------------------
void Voxcel_FriendsOfFriends_omp()
{
    int p;
    double LINKLENGHT = LINKFACTOR / cbrt(NMEAN);
    float  LINKGRID   = LINKLENGHT / LGRID;
    int    nsearch    = (int)LINKGRID + 1;

    fprintf(stdout, "\n Comienza busqueda VoxcelFoF\n"); fflush(stdout);
    double t_ini = omp_get_wtime();

    GrpID = (int*)malloc(NG * NG * NG * sizeof(int));
    for (p = 0; p < NG * NG * NG; p++)
        GrpID[p] = -1;

    int Ngroup = -1;
    for (p = 0; p < NG * NG * NG; p++) {
        if (Cabecera[p] == -1) continue; // empty grid
        if (GrpID[p] != -1)   continue;  // already assigned

        Ngroup++;

        std::queue<int> Q;
        Q.push(p);
        GrpID[p] = Ngroup;

        Gvoxcel.push_back(groups());
        Gvoxcel[Ngroup].Mem.push_back(p);

        while (!Q.empty()) {
            int qsize = (int)Q.size();

            // drain current level
            std::vector<int> current_level(qsize);
            for (int i = 0; i < qsize; ++i) {
                current_level[i] = Q.front();
                Q.pop();
            }

            // thread-private fan-out
            std::vector<std::vector<int>> Q_private(omp_get_max_threads());

            #pragma omp parallel for schedule(dynamic)
            for (int idx = 0; idx < qsize; idx++) {
                int pp = current_level[idx];

                int kc = pp % NG;
                int jc = (pp / NG) % NG;
                int ic = pp / (NG * NG);

                for (int it = ic - nsearch; it <= ic + nsearch; it++) {
                    int shifti = it - ic;
                    int ii = iper(it);
                    for (int jt = jc - nsearch; jt <= jc + nsearch; jt++) {
                        int shiftj = jt - jc;
                        int jj = iper(jt);
                        for (int kt = kc - nsearch; kt <= kc + nsearch; kt++) {
                            int shiftk = kt - kc;
                            int kk = iper(kt);

                            if (!is_touched(shifti, shiftj, shiftk, LINKGRID)) continue;

                            int l = (ii * NG + jj) * NG + kk;
                            if (Cabecera[l] == -1) continue;  // empty
                            if (GrpID[l] != -1)   continue;  // already assigned

                            int  tid     = omp_get_thread_num();
                            bool do_push = false;

                            #pragma omp critical
                            {
                                if (GrpID[l] == -1) {
                                    GrpID[l] = Ngroup;
                                    Gvoxcel[Ngroup].Mem.push_back(l);
                                    do_push = true;
                                }
                            }

                            if (do_push) Q_private[tid].push_back(l);
                        }
                    }
                }
            }

            // push next frontier
            for (auto& local_q : Q_private) {
                for (int item : local_q) Q.push(item);
            }
        }
    }

    double t_fin = omp_get_wtime();
    fprintf(stdout, " | FoF on voxcels -> tiempo = %f min. ng %zu\n",
            (t_fin - t_ini) / 60.0, Gvoxcel.size());
}

// ------------------------------------------------------------
// Voxel-level FoF (serial BFS on voxel graph)
// ------------------------------------------------------------
void Voxcel_FriendsOfFriends()
{
    double LINKLENGHT = LINKFACTOR / cbrt(NMEAN);
    float  LINKGRID   = LINKLENGHT / LGRID;
    int    nsearch    = (int)LINKGRID + 1;

    fprintf(stdout, "\n Comienza busqueda VoxcelFoF\n"); fflush(stdout);
    double t_ini = omp_get_wtime();

    GrpID = (int*)malloc(NG * NG * NG * sizeof(int));
    for (int p = 0; p < NG * NG * NG; p++)
        GrpID[p] = -1;

    int Ngroup = -1;
    for (int p = 0; p < NG * NG * NG; p++) {
        if (Cabecera[p] == -1) continue; // empty grid
        if (GrpID[p] != -1)   continue;  // already assigned

        Ngroup++;

        std::queue<int> Q;
        Q.push(p);
        GrpID[p] = Ngroup;

        groups test_group{};
        test_group.Mem.push_back(p);

        do {
            int pp = Q.front(); Q.pop();

            int kc = pp % NG;
            int jc = (pp / NG) % NG;
            int ic = pp / (NG * NG);

            for (int it = ic - nsearch; it <= ic + nsearch; it++) {
                int shifti = it - ic;
                int ii = iper(it);
                for (int jt = jc - nsearch; jt <= jc + nsearch; jt++) {
                    int shiftj = jt - jc;
                    int jj = iper(jt);
                    for (int kt = kc - nsearch; kt <= kc + nsearch; kt++) {
                        int shiftk = kt - kc;
                        int kk = iper(kt);

                        if (!is_touched(shifti, shiftj, shiftk, LINKGRID)) continue;

                        int l = (ii * NG + jj) * NG + kk;
                        int next = Cabecera[l];
                        if (next == -1) continue;
                        if (GrpID[l] != -1) continue;

                        Q.push(l);
                        GrpID[l] = Ngroup;
                        test_group.Mem.push_back(l);
                    }
                }
            }
        } while (!Q.empty());

        Gvoxcel.push_back(test_group);
    }

    double t_fin = omp_get_wtime();
    fprintf(stdout, " | FoF on voxcels -> tiempo = %f min. ng %zu\n",
            (t_fin - t_ini) / 60.0, Gvoxcel.size());
}

// ------------------------------------------------------------
// Particle-level FoF inside each voxel component
// (fixed COM: include seed + unwrap to fixed seed reference)
// ------------------------------------------------------------
void FriendsOfFriends()
{
    double LINKLENGHT = LINKFACTOR / cbrt(NMEAN);
    float  LINKGRID   = LINKLENGHT / LGRID;
    int    nsearch    = (int)LINKGRID + 1;

    fprintf(stdout, "\n Comienza busqueda FoF\n"); fflush(stdout);
    double t_ini = omp_get_wtime();

    std::vector<std::vector<struct groups>> local_G(omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic)
    for (int grp = 0; grp < (int)Gvoxcel.size(); grp++) {
        for (int voxcen = 0; voxcen < (int)Gvoxcel[grp].Mem.size(); voxcen++) {

            int lc    = Gvoxcel[grp].Mem[voxcen];
            int start = Cabecera[lc];
            if (start == -1) continue; // empty cell

            int p = start;
            do {
                if (Tracer[p].GrpID != -1) { // already assigned to a particle-group
                    p = Linklist[p];
                    continue;
                }

                std::queue<int> Q;
                groups test_group{};       // zero sums

                // seed
                Q.push(p);
                test_group.Mem.push_back(p);
                Tracer[p].GrpID = 0;

                // fixed reference for unwrapping
                double xref[3] = {
                    Tracer[p].Pos[0],
                    Tracer[p].Pos[1],
                    Tracer[p].Pos[2]
                };

                // include SEED in sums
                for (int k = 0; k < 3; ++k) {
                    test_group.Pos[k] += xref[k];
                    test_group.Vel[k] += Tracer[p].Vel[k];
                }

                // BFS grow
                do {
                    int pp = Q.front(); Q.pop();

                    double xc[3] = {
                        Tracer[pp].Pos[0],
                        Tracer[pp].Pos[1],
                        Tracer[pp].Pos[2]
                    };

                    int ic = (int)(xc[0] / LGRID);
                    int jc = (int)(xc[1] / LGRID);
                    int kc = (int)(xc[2] / LGRID);

                    for (int it = ic - nsearch; it <= ic + nsearch; ++it) {
                        int shifti = it - ic;
                        int ii = iper(it);
                        for (int jt = jc - nsearch; jt <= jc + nsearch; ++jt) {
                            int shiftj = jt - jc;
                            int jj = iper(jt);
                            for (int kt = kc - nsearch; kt <= kc + nsearch; ++kt) {
                                int shiftk = kt - kc;
                                int kk = iper(kt);

                                if (!is_touched(shifti, shiftj, shiftk, LINKGRID)) continue;

                                int l = (ii * NG + jj) * NG + kk;
                                int next = Cabecera[l];
                                if (next == -1) continue;

                                do {
                                    next = Linklist[next];
                                    if (Tracer[next].GrpID != -1) continue;

                                    // distance check around current center pp
                                    double xt[3], dx[3];
                                    for (int k = 0; k < 3; ++k) {
                                        xt[k] = Tracer[next].Pos[k];
                                        dx[k] = xc[k] - xt[k];
                                        if (PERIODIC) {
                                            if (dx[k] >  0.5 * LBOX) dx[k] -= LBOX;
                                            if (dx[k] < -0.5 * LBOX) dx[k] += LBOX;
                                        }
                                    }
                                    double dist = sqrt(dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2]);

                                    if (dist <= LINKLENGHT) {
                                        Q.push(next);
                                        Tracer[next].GrpID = 0;
                                        test_group.Mem.push_back(next);

                                        // add to sums, unwrapped to seed reference
                                        for (int k = 0; k < 3; ++k) {
                                            double xn = Tracer[next].Pos[k];
                                            double dref = xn - xref[k];
                                            if (PERIODIC) {
                                                if (dref >  0.5 * LBOX) xn -= LBOX;
                                                if (dref < -0.5 * LBOX) xn += LBOX;
                                            }
                                            test_group.Pos[k] += xn;
                                            test_group.Vel[k] += Tracer[next].Vel[k];
                                        }
                                    }
                                } while (next != Cabecera[l]);
                            }
                        }
                    }
                } while (!Q.empty());

                int tid = omp_get_thread_num();
                local_G[tid].push_back(test_group);

                p = Linklist[p];
            } while (p != start);
        }
    }

    // merge per-thread vectors
    for (auto& local_fof : local_G) {
        for (auto& item : local_fof) G.push_back(item);
    }

    // assign final group ids
    int ngr = (int)G.size();
    for (int i = 0; i < ngr; i++) {
        int nmem = (int)G[i].Mem.size();
        for (int j = 0; j < nmem; j++)
            Tracer[G[i].Mem[j]].GrpID = i;
    }

    double t_fin = omp_get_wtime();
    fprintf(stdout, " | FoF -> tiempo = %f min.\n", (t_fin - t_ini) / 60.0);
}
