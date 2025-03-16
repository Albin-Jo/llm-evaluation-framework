import React from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  BarChart,
  Database,
  Brain,
  GitCompare,
  MessageSquare,
  ArrowRight,
  BarChart2
} from 'lucide-react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { getDatasets } from '@/api/datasets';
import { DevelopmentNotice } from '@/components/development-notice';

export function DashboardPage() {
  const { data: datasetsData } = useQuery({
    queryKey: ['datasets', 1, ''],
    queryFn: () => getDatasets({ page: 1, limit: 5 }),
  });

  const datasets = datasetsData?.items || [];
  const datasetsCount = datasetsData?.total || 0;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            Last 7 Days
          </Button>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Datasets</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{datasetsCount}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Evaluations</CardTitle>
            <BarChart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-muted">-</div>
            <p className="text-xs text-muted-foreground">Coming soon</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Micro Agents</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-muted">-</div>
            <p className="text-xs text-muted-foreground">Coming soon</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Prompts</CardTitle>
            <MessageSquare className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-muted">-</div>
            <p className="text-xs text-muted-foreground">Coming soon</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card className="col-span-2">
          <CardHeader>
            <CardTitle>Recent Datasets</CardTitle>
            <CardDescription>
              Recently created and updated datasets
            </CardDescription>
          </CardHeader>
          <CardContent>
            {datasets.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-6 text-center">
                <Database className="h-10 w-10 text-muted mb-2" />
                <h3 className="text-lg font-medium">No datasets yet</h3>
                <p className="text-sm text-muted-foreground mt-1">
                  Create your first dataset to get started
                </p>
                <Button className="mt-4" asChild>
                  <Link to="/datasets/new">
                    Create Dataset
                  </Link>
                </Button>
              </div>
            ) : (
              <div className="space-y-2">
                {datasets.map((dataset) => (
                  <div
                    key={dataset.id}
                    className="flex items-center justify-between p-3 border rounded-md hover:bg-muted/50 transition-colors"
                  >
                    <div className="flex items-center space-x-4">
                      <Database className="h-6 w-6 text-muted-foreground" />
                      <div>
                        <p className="font-medium">{dataset.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {dataset.type} • {dataset.row_count || 'Unknown'} rows
                        </p>
                      </div>
                    </div>
                    <Button variant="ghost" size="sm" asChild>
                      <Link to={`/datasets/${dataset.id}`}>
                        <ArrowRight className="h-4 w-4" />
                      </Link>
                    </Button>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
          <CardFooter>
            <Button variant="outline" className="w-full" asChild>
              <Link to="/datasets">View All Datasets</Link>
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Evaluations</CardTitle>
            <CardDescription>
              Latest evaluation runs
            </CardDescription>
          </CardHeader>
          <CardContent>
            <DevelopmentNotice />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Performance Metrics</CardTitle>
            <CardDescription>
              Overall evaluation performance
            </CardDescription>
          </CardHeader>
          <CardContent className="h-80 flex items-center justify-center">
            <div className="flex flex-col items-center text-center">
              <BarChart2 className="h-16 w-16 text-muted opacity-50" />
              <p className="mt-4 text-lg font-medium">Performance data coming soon</p>
              <p className="text-sm text-muted-foreground mt-1">
                This feature will be available once evaluations are implemented
              </p>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Comparisons</CardTitle>
            <CardDescription>
              Latest model comparison results
            </CardDescription>
          </CardHeader>
          <CardContent>
            <DevelopmentNotice title="Comparisons Coming Soon" message="The comparison feature is under development and will be available in a future release." />
          </CardContent>
          <CardFooter>
            <Button variant="outline" className="w-full" disabled>
              View All Comparisons
            </Button>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}