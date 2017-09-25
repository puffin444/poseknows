set(0,'DefaultFigureColor',[1 1 1])

doPlot('/nfs/hn48/jcwalker/caffe_lstm/finalRelease/compare/', 100, 'r', '-',2)

hold on

legend('Pose-VAE') 

title('UCF101 Minimum Euclidean Distance')

xlabel('Number of Samples');
ylabel('Minimum Euclidean Distance')

