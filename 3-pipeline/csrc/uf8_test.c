extern int uf8_decode(int fl);
extern int uf8_encode(int value);

int main()
{
    int previous_value = -1;
    
    for (int i = 0; i < 256; i++) {
        int fl = i;
        int value = uf8_decode(fl);
        int fl2 = uf8_encode(value);

        if (fl != fl2 || value <= previous_value) {
            *(int *)(4) = 0; 
            return 0;
        }

        previous_value = value;
    }

    *(int *)(4) = 1;
    return 0;
}