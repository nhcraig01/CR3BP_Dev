function h = plotBrkStab(BrkVals, opts)


% ------ Parse -------
arguments
    BrkVals (:,2) double
    opts.Color = [0,0,0]
    opts.Linewidth (1,1) double {mustBePositive} = 1
    opts.Marker (1,1) string = "+"
    opts.Labels (1,1) logical = true
    opts.Hold (1,1) string {mustBeMember(opts.Hold,["auto","on","off"])} = "auto"
end

% -------- Plot line --------
% ax = gca;
% restoreHold = false;
% if opts.Hold == "auto"
%     restoreHold = ~isHold(ax);
%     hold(ax,'on');
% elseif opts.Hold == "on"
%     hold(ax,'on');
% else
%     hold(ax,'off');
% end

h = struct('line',[],'labels',[]);

h.line = plot(BrkVals(:,1),BrkVals(:,2),...
    'Color',opts.Color, 'LineWidth',opts.Linewidth,...
    'Marker',opts.Marker,'LineStyle','--');
if opts.Labels
    h.labels = text(BrkVals(:,1),BrkVals(:,2),string(1:length(BrkVals(:,1))));
end


hold on;
fplot(@(a) -2.*a-2,'LineWidth',1.5); % Tangent
fplot(@(a) 2.*a-2,'LineWidth',1.5); % Period doubling
fplot(@(a) a+1,'LineWidth',1.5); % Period Tripling
fplot(2,'LineWidth',1.5); % Period Quadrupling
% axis equal;
xlabel('\alpha');
ylabel('\beta');
legend('Orbit Family','Tangent','P2','P3',...
    'P4');


% if restoreHold, hold(ax,'off'); end
end
