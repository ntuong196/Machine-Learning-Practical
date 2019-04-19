function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;
  
  %%% TODO: Fill in the rest of this function... 
  classes = unique(Y); 
  figure(); 
  scatter(X(Y==classes(1),1),  X(Y==classes(1),2), 'filled','d');
  hold on 
  scatter(X(Y==classes(2),1),  X(Y==classes(2),2));
  hold on 
  plot_x = [min(X(:,1))-2,  max(X(:,2))+2];
  plot_y = (-1./obj.wts(3)).*(obj.wts(2).*plot_x + obj.wts(1));
  % Plot, and adjust axes for better viewing
  plot(plot_x, plot_y);