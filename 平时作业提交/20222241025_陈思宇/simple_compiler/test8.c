// test8.c
int a;
int b;
a = 5;
b = 10;
if (a < b) {
    while (a < b) {
        a = a + 1;
    }
} else {
    a = b - a;
}
