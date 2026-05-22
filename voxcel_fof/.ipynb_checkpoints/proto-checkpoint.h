
void Sort_part(void);
int peano_hilbert_key(int x, int y, int );
void Grid(void);
int  iper(int);
void FreeGrid(void);
void ReadTracers(void); 
int  CountLines(const char *);
void Time(clock_t ti, const char* msg);
void Progress(int, int);
void Voxcel_FriendsOfFriends(void); 
void FriendsOfFriends(void); 
void WriteGroups(void);
static inline bool is_touched(int i, int j, int k, float rad) 
{
    float dx, dy, dz;

    if (i >= 1) {
        dx = i - 1;
    } else if (i <= -1) {
        dx = -i - 1;
    } else {
        dx = 0.0f;
    }

    if (j >= 1) {
        dy = j - 1;
    } else if (j <= -1) {
        dy = -j - 1;
    } else {
        dy = 0.0f;
    }

    if (k >= 1) {
        dz = k - 1;
    } else if (k <= -1) {
        dz = -k - 1;
    } else {
        dz = 0.0f;
    }

    float D2 = dx*dx + dy*dy + dz*dz;
    return D2 <= rad * rad;
}
