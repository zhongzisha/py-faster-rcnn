
nets = {'zf','vgg16','d_zf','d_vgg16','d_zf1','d_vgg161','d_vgg162'};
colors = VOClabelcolormap(20); %rand(20,3);
legend_strs = cell(1, length(nets));

VOCinit; 
for i=1:length(nets)
tic,[rec,prec,ap] = VOCevaldet(VOCopts,nets{i},'car',0);
legend_strs{i} = sprintf('%s(%.4f)',nets{i},ap);
plot(rec,prec,'-','Color',colors(i,:));hold on
end

grid;
xlabel 'recall'
ylabel 'precision'
legend(legend_strs,'Location','SouthWest','Interpreter', 'none')


















