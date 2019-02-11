CROSS_VALL_DNN
z = [89.818   90.679   89.520   91.520   91.547   91.543   91.562
90.117   91.085   89.451   91.508   91.497   91.554   91.477
89.938   91.244   89.491   91.700   91.637   91.641   91.536
89.248   90.604   88.418   91.020   91.246   91.205   91.103
89.784   90.537   88.776   91.089   91.308   91.277   91.193
89.650   90.592   88.622   91.082   91.250   91.232   91.292];
for i=1:6
  plot(1:7,z(i,:), 'linewidth', 3, 'markersize', 40, '.-')
  hold on
end
set(gca, 'fontsize', 20)
set(legend('200 150 100 100','200 100 50','200 100','100 50 25 25','100 50 25','100 50'), 'fontsize',20, 'location', 'southeast')
set(gca, 'xtick', 1:7)
set(gca, 'xticklabel', ['SGD';'RMSprop';'Adagrad';'Adadelta';'Adam';'Adamax';'Nadam'])
set(xlabel('Optimizers'), 'fontsize', 20)
set(ylabel('Accuracy (%)'), 'fontsize', 20)

CROSS_VAL_SVM - sigmoid
z = [16.667   29.791   16.667 #auto
16.667   26.067   16.667
16.667   21.711   16.667];

z = [16.667   30.556892954581927   16.667 #scale
16.667   26.067   16.667
16.667   19.541   16.667];
for i=1:3
  plot(1:3,z(i,:), 'linewidth', 3, 'markersize', 40, '.-')
  hold on
end
set(gca, 'fontsize', 20)
set(legend('0.001', '1', '30'), 'fontsize',20, 'location', 'northeast')
set(gca, 'xtick', 1:7)
set(gca, 'xticklabel', [-25,0,25])
set(xlabel('coef0'), 'fontsize', 20)
set(ylabel('Accuracy (%)'), 'fontsize', 20)

LINEAR
z = [0.37312429786551116 0.3922925694110095 0.400392934] .* 100;
plot(1:3,z, 'linewidth', 3, 'markersize', 40, '.-')
set(gca, 'fontsize', 20)
set(gca, 'xtick', 1:3)
set(gca, 'xticklabel', [0.001, 1, 30])
set(xlabel('C'), 'fontsize', 20)
set(ylabel('Accuracy (%)'), 'fontsize', 20)

RBF
z = [31.1286310383566   37.829   48.612   62.869   70.211   72.232   73.328
32.651259829882845   41.913   57.487   68.741   75.790   77.107   77.592];
for i=1:2
  plot(1:7,z(i,:), 'linewidth', 3, 'markersize', 40, '.-')
  hold on
end
set(gca, 'fontsize', 20)
set(legend('auto', 'scale'), 'fontsize',20, 'location', 'southeast')
set(gca, 'xtick', 1:7)
set(gca, 'xticklabel', [0.001, 0.01, 0.1, 1, 10, 20, 30])
set(xlabel('C'), 'fontsize', 20)
set(ylabel('Accuracy (%)'), 'fontsize', 20)
