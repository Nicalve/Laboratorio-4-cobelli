%% Curvas difracciín con/sin ruido, con/sin saturación

% Longitud de onda: 632.8nm
L=632.8e-9;

%tamaño de rendijas en m
slits=10e-6;

%distancia de la rendija a la pantalla en m
z=0.10;

%coordenada en la pantalla
x=-35e-3:0.069e-3:35e-3;

A=100 %intensidad del patrón
B=50   %intensidad del fondo variable 
C=30  %intensidad del fondo fijo
D=5  %nivel de ruido

dif=A*(sin(pi*slits*x/L/z)).^2./(pi*slits*x/L/z).^2;

fondo=B*x+C;

ruido=rand(1,1015)*D;
todo=dif+fondo+ruido;
image=uint8(todo);

figure(98),
subplot(2,1,1)
plot(x,todo),ylabel('Intensidad [u.arb.]')
subplot(2,1,2)
plot(x,image),xlabel('Posición [mm]'),ylabel('Escala de grises 8 bit')
