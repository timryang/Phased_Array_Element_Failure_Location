nof_1 = imread('Patterns\Test\14dB_50Its_M5_Size8_Phase_Test\image_00_1.png');
nof_2 = imread('Patterns\Test\14dB_50Its_M5_Size8_Phase_Test\image_00_2.png');
nof_3 = imread('Patterns\Test\14dB_50Its_M5_Size8_Phase_Test\image_00_3.png');

figure('Position',[100,100,1000,300]); hold on;
colormap gray
subplot(1,3,1)
imagesc(nof_1)
axis off
axis square
title('No Failure, SNR = 14 dB')
subplot(1,3,2)
imagesc(nof_2)
axis off
axis square
title('No Failure, SNR = 14 dB')
subplot(1,3,3)
imagesc(nof_3)
axis off
axis square
title('No Failure, SNR = 14 dB')

%%

f_1 = imread('Patterns\Test\14dB_50Its_M5_Size8_Phase_Test\image_02_101.png');
f_2 = imread('Patterns\Test\14dB_50Its_M5_Size8_Phase_Test\image_02_102.png');
f_3 = imread('Patterns\Test\14dB_50Its_M5_Size8_Phase_Test\image_02_103.png');

figure('Position',[100,100,1000,300]); hold on;
colormap gray
subplot(1,3,1)
imagesc(f_1)
axis off
axis square
title('Failure State 2, SNR = 14 dB')
subplot(1,3,2)
imagesc(f_2)
axis off
axis square
title('Failure State 2, SNR = 14 dB')
subplot(1,3,3)
imagesc(f_3)
axis off
axis square
title('Failure State 2, SNR = 14 dB')

%%

amp_1 = imread('16dB_M5_Size8_Amp\image_00_1.png');
amp_2 = imread('16dB_M5_Size8_Amp\image_12_37.png');
amp_3 = imread('16dB_M5_Size8_Amp\image_24_73.png');

figure('Position',[100,100,1000,300]); hold on;
colormap gray
subplot(1,3,1)
imagesc(10*log10(double(amp_1)/255))
axis off
axis square
title('No Failure, SNR = 16 dB')
subplot(1,3,2)
imagesc(10*log10(double(amp_2)/255))
axis off
axis square
title('Failure State 12, SNR = 16 dB')
subplot(1,3,3)
imagesc(10*log10(double(amp_3)/255))
axis off
axis square
title('Failure State 24, SNR = 16 dB')