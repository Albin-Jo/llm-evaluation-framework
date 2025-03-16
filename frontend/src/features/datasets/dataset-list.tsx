import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import {
  Plus,
  Search,
  FileText,
  Trash2,
  Edit,
  Eye,
  Database,
  ArrowUpDown,
  Filter
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { getDatasets, Dataset } from '@/api/datasets';
import { formatDate } from '@/lib/utils';

export function DatasetList() {
  const [searchQuery, setSearchQuery] = useState('');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;

  const { data, isLoading, error } = useQuery({
    queryKey: ['datasets', currentPage, searchQuery],
    queryFn: () => getDatasets({
      page: currentPage,
      limit: itemsPerPage,
      search: searchQuery || undefined
    })
  });

  const datasets = data?.items || [];
  const totalItems = data?.total || 0;
  const totalPages = Math.ceil(totalItems / itemsPerPage);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setCurrentPage(1); // Reset to first page on new search
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Error Loading Datasets</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-destructive">Failed to load datasets. Please try again later.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">Datasets</h1>
        <Button asChild>
          <Link to="/datasets/new">
            <Plus className="mr-2 h-4 w-4" />
            New Dataset
          </Link>
        </Button>
      </div>

      <Card>
        <CardHeader className="pb-3">
          <div className="flex justify-between items-center">
            <CardTitle>Available Datasets</CardTitle>
            <form onSubmit={handleSearch} className="flex w-full max-w-sm items-center space-x-2">
              <Input
                type="search"
                placeholder="Search datasets..."
                value={searchQuery}
                onChange={handleSearchChange}
                className="max-w-xs"
              />
              <Button type="submit" size="sm" variant="secondary">
                <Search className="h-4 w-4" />
              </Button>
            </form>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b bg-muted/50">
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    <div className="flex items-center">
                      Name
                      <ArrowUpDown className="ml-2 h-4 w-4" />
                    </div>
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">
                    <div className="flex items-center">
                      Type
                      <Filter className="ml-2 h-4 w-4" />
                    </div>
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">Created</th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-muted-foreground">Rows</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-muted-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {isLoading ? (
                  Array(5).fill(0).map((_, i) => (
                    <tr key={i} className="border-b">
                      <td className="px-4 py-3">
                        <div className="h-5 w-48 bg-muted animate-pulse rounded"></div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="h-5 w-24 bg-muted animate-pulse rounded"></div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="h-5 w-32 bg-muted animate-pulse rounded"></div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="h-5 w-16 bg-muted animate-pulse rounded"></div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="h-5 w-24 bg-muted animate-pulse rounded"></div>
                      </td>
                    </tr>
                  ))
                ) : datasets.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-4 py-6 text-center text-muted-foreground">
                      <Database className="mx-auto h-10 w-10 mb-2 opacity-20" />
                      <p>No datasets found</p>
                      {searchQuery && (
                        <p className="text-sm mt-1">
                          Try adjusting your search or{" "}
                          <Button variant="link" className="h-auto p-0" onClick={() => setSearchQuery("")}>
                            clear filters
                          </Button>
                        </p>
                      )}
                    </td>
                  </tr>
                ) : (
                  datasets.map((dataset: Dataset) => (
                    <tr key={dataset.id} className="border-b">
                      <td className="px-4 py-3">
                        <div className="font-medium">{dataset.name}</div>
                        {dataset.description && (
                          <div className="text-sm text-muted-foreground">
                            {dataset.description.length > 60
                              ? `${dataset.description.substring(0, 60)}...`
                              : dataset.description}
                          </div>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <span className="inline-flex items-center rounded-md bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-700/10 dark:bg-blue-400/10 dark:text-blue-400 dark:ring-blue-400/20">
                          {dataset.type}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-muted-foreground">
                        {dataset.created_at ? formatDate(dataset.created_at) : 'N/A'}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {dataset.row_count ?? 'N/A'}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex justify-end space-x-2">
                          <Button variant="ghost" size="icon" asChild>
                            <Link to={`/datasets/${dataset.id}/preview`}>
                              <Eye className="h-4 w-4" />
                              <span className="sr-only">Preview</span>
                            </Link>
                          </Button>
                          <Button variant="ghost" size="icon" asChild>
                            <Link to={`/datasets/${dataset.id}`}>
                              <FileText className="h-4 w-4" />
                              <span className="sr-only">View</span>
                            </Link>
                          </Button>
                          <Button variant="ghost" size="icon" asChild>
                            <Link to={`/datasets/${dataset.id}/edit`}>
                              <Edit className="h-4 w-4" />
                              <span className="sr-only">Edit</span>
                            </Link>
                          </Button>
                          <Button variant="ghost" size="icon">
                            <Trash2 className="h-4 w-4 text-destructive" />
                            <span className="sr-only">Delete</span>
                          </Button>
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
        {totalPages > 1 && (
          <CardFooter className="flex justify-between items-center border-t px-6 py-3">
            <div className="text-sm text-muted-foreground">
              Showing <span className="font-medium">{Math.min((currentPage - 1) * itemsPerPage + 1, totalItems)}</span> to{" "}
              <span className="font-medium">{Math.min(currentPage * itemsPerPage, totalItems)}</span> of{" "}
              <span className="font-medium">{totalItems}</span> results
            </div>
            <div className="flex space-x-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => handlePageChange(currentPage - 1)}
                disabled={currentPage === 1}
              >
                Previous
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => handlePageChange(currentPage + 1)}
                disabled={currentPage >= totalPages}
              >
                Next
              </Button>
            </div>
          </CardFooter>
        )}
      </Card>
    </div>
  );
}