% RRT* algorithm in 2D with collision avoidance.
% 
% Author: Sai Vemprala
% 
% nodes:    Contains list of all explored nodes. Each node contains its
%           coordinates, cost to reach and its parent.
% 
% Brief description of algorithm: 
% 1. Pick a random node q_rand.
% 2. Find the closest node q_near from explored nodes to branch out from, towards
%    q_rand.
% 3. Steer from q_near towards q_rand: interpolate if node is too far away, reach
%    q_new. Check that obstacle is not hit.
% 4. Update cost of reaching q_new from q_near, treat it as Cmin. For now,
%    q_near acts as the parent node of q_new.
% 5. From the list of 'visited' nodes, check for nearest neighbors with a 
%    given radius, insert in a list q_nearest.
% 6. In all members of q_nearest, check if q_new can be reached from a
%    different parent node with cost lower than Cmin, and without colliding
%    with the obstacle. Select the node that results in the least cost and 
%    update the parent of q_new.
% 7. Add q_new to node list.
% 8. Continue until maximum number of nodes is reached or goal is hit.

clearvars
close all


x_max = 100;
y_max = 100;
obstacle = [500,150,200,200];

% poly = { [500, 150; 500, 350; 700, 350; 700, 150],...
%          [500, 450; 500 650 ; 700 650 ; 700 450],...
%          [200, 450; 100 300 ; 280 320],...
%          [200, 600; 450 600 ; 450 530; 200 530],...
%          };

O1 = [50, 15; 50, 35; 70, 35; 70, 15];
O2 = [50, 45; 50 65 ; 70 65 ; 70 45];
O3 = [20, 45; 10 30 ; 28 32];
O4 = [20, 60; 45 60 ; 45 53; 20 53];

poly = { [50, 15; 50, 35; 70, 35; 70, 15],...
         [50, 45; 50 65 ; 70 65 ; 70 45],...
         [20, 45; 10 30 ; 28 32],...
         [20, 60; 45 60 ; 45 53; 20 53],...
         };    
    
EPS = 1;
numNodes = 500;  %     

q_start.coord = [5 60];
q_start.cost = 0;
q_start.parent = 0;
q_goal.coord = [80 30];
q_goal.cost = 0;

nodes(1) = q_start;
figure(1)
axis([0 x_max 0 y_max])
% rectangle('Position',obstacle,'FaceColor',[0 .5 .5])
plot_obstacle_poly(gca, poly);
plot(80, 30, 'go', 'MarkerSize',10, 'MarkerFaceColor','g');
plot(5,60, 'ro', 'MarkerSize',10, 'MarkerFaceColor','r');

% p2=[13.04,53.09];
% p3=[32.73,52.4];
% p4=[78.56,32.28];

% p2=[17.79,53.52];
% p3=[67.95,38.75];
% p4=[75.17,29.01];

% plot(13.04,53.09, 'ro', 'MarkerSize',10, 'MarkerFaceColor','r');
% plot(32.73,52.4, 'ro', 'MarkerSize',10, 'MarkerFaceColor','r');
% plot(78.56,32.28, 'ro', 'MarkerSize',10, 'MarkerFaceColor','r');



hold on

for i = 1:1:numNodes
    
    t1 = tic;
    K1 = rand(i) ;   
    thenorm1 = norm(K1) ;
    t1 = toc( t1 );
    
    q_rand = [floor(rand(1)*x_max) floor(rand(1)*y_max)];
    plot(q_rand(1), q_rand(2), 'x', 'Color',  [0 0.4470 0.7410])
    
    % Break if goal node is already reached
    for j = 1:1:length(nodes)
        if nodes(j).coord == q_goal.coord
            break
        end
    end
    
    % Pick the closest node from existing list to branch out from
    ndist = [];
    for j = 1:1:length(nodes)
        n = nodes(j);
        tmp = dist(n.coord, q_rand);
        ndist = [ndist tmp];
    end
    [val, idx] = min(ndist);
    q_near = nodes(idx);
    
    q_new.coord = steer(q_rand, q_near.coord, val, EPS);
%     if noCollision(q_rand, q_near.coord, obstacle)
    if (chk_collision([nodes(j).coord; q_new.coord],poly) == 0) 
        line([q_near.coord(1), q_new.coord(1)], [q_near.coord(2), q_new.coord(2)], 'Color', 'k', 'LineWidth', 2);
        drawnow
        hold on
        q_new.cost = dist(q_new.coord, q_near.coord) + q_near.cost;
        
        % Within a radius of r, find all existing nodes
        q_nearest = [];
        r = 60;
        neighbor_count = 1;
        for j = 1:1:length(nodes)
%             if noCollision(nodes(j).coord, q_new.coord, obstacle) && dist(nodes(j).coord, q_new.coord) <= r
            if (chk_collision([nodes(j).coord; q_new.coord],poly) == 0) && dist(nodes(j).coord, q_new.coord) <= r

                q_nearest(neighbor_count).coord = nodes(j).coord;
                q_nearest(neighbor_count).cost = nodes(j).cost;
                neighbor_count = neighbor_count+1;
            end
        end
        
        % Initialize cost to currently known value
        q_min = q_near;
        C_min = q_new.cost;
        
        % Iterate through all nearest neighbors to find alternate lower
        % cost paths
        
        for k = 1:1:length(q_nearest)
            if noCollision(q_nearest(k).coord, q_new.coord, obstacle) && q_nearest(k).cost + dist(q_nearest(k).coord, q_new.coord) < C_min
                q_min = q_nearest(k);
                C_min = q_nearest(k).cost + dist(q_nearest(k).coord, q_new.coord);
                line([q_min.coord(1), q_new.coord(1)], [q_min.coord(2), q_new.coord(2)], 'Color', 'g');                
                hold on
            end
        end
        
        % Update parent to least cost-from node
        for j = 1:1:length(nodes)
            if nodes(j).coord == q_min.coord
                q_new.parent = j;
            end
        end
        
        % Append to nodes
        nodes = [nodes q_new];
    end
    
   result1(i) = thenorm1 ;   % store your result for final average 
    elapsed_time1(i) = t1 ;   % store the time elapsed for the run 

end

% get avarages 
result_avg1 = mean(result1) ;
time_avg1 = mean(elapsed_time1) ;

D = [];
for j = 1:1:length(nodes)
    
    t2 = tic;
    K2 = rand(j) ;   
    thenorm2 = norm(K2) ;
    t2 = toc( t2 );
    
    tmpdist = dist(nodes(j).coord, q_goal.coord);
    D = [D tmpdist];
    
   result2(j) = thenorm2 ;   % store your result for final average 
    elapsed_time2(j) = t2 ;   % store the time elapsed for the run 

end

% get avarages 
result_avg2 = mean(result2) ;
time_avg2 = mean(elapsed_time2) ;

% Search backwards from goal to start to find the optimal least cost path
[val, idx] = min(D);
q_final = nodes(idx);
q_goal.parent = idx;
q_end = q_goal;
nodes = [nodes q_goal];
while q_end.parent ~= 0
    start = q_end.parent;
    line([q_end.coord(1), nodes(start).coord(1)], [q_end.coord(2), nodes(start).coord(2)], 'Color', 'r', 'LineWidth', 3);
    
    dis =  sqrt( (nodes(start).coord(1)-nodes(start).coord(2))^2 + ((q_end.coord(1))-q_end.coord(2))^2 )  ;
    dis1 =  sqrt( (nodes(start).coord(1)-q_end.coord(1))^2 + (nodes(start).coord(2)-q_end.coord(2))^2 )  ;

    sum(dis)
    sum(dis1)
    
    hold on
    q_end = nodes(start);
end


