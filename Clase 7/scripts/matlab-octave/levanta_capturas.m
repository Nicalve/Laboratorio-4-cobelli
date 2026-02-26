
ggg=imread('captura4.png');
gg1=double(ggg(:,:,1));

figure(57),surf(gg1),shading flat, view([0 -90]),daspect([1 1 1]),colorbar

g1d=gg1(480,:);figure(23),plot(g1d)