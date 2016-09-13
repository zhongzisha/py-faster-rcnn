
nets = {'zf','vgg16','d_zf','d_vgg16','d_zf1','d_vgg161'};
colors = 'rgby';
legend_strs = cell(1, length(nets));

VOCinit;
ind = 1;
for i=1:2:length(nets)
tic,[rec,prec,ap] = VOCevaldet(VOCopts,nets{i},'car',0);
legend_strs{i} = sprintf('%s(%.4f)',nets{i},ap);
plot(rec,prec,'-','Color',colors(ind));hold on
tic,[rec,prec,ap] = VOCevaldet(VOCopts,nets{i+1},'car',0);
legend_strs{i+1} = sprintf('%s(%.4f)',nets{i+1},ap);
plot(rec,prec,'-.','Color',colors(ind));hold on
ind = ind + 1;
end

grid;
xlabel 'recall'
ylabel 'precision'
legend(legend_strs,'Location','SouthWest','Interpreter', 'none')


















