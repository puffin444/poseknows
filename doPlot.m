function doPlot(thePath, numSamp, color, style, lwidth)
theDiffs = [];

theFiles = dir([thePath 'diff*.mat'])

for i=1:length(theFiles)
   temp = hdf5read([thePath theFiles(i).name], '/diff');
   theDiffs = [theDiffs;reshape(temp, [32 numSamp])];
end

thePlot = [];
theAxis = [];
for i = 1:numSamp
theAxis = [theAxis; i];
thePlot = [thePlot; mean(min(theDiffs(:,1:i),[],2))];
end

if(numSamp < 100)
thePlot = ones(100,1)*thePlot(1)
theAxis = 1:100

end


plot(theAxis, thePlot,'Color', color, 'LineStyle', style, 'LineWidth', lwidth)

end
