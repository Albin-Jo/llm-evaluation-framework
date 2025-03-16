import React from 'react';
import { useParams, Link, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { ChevronLeft, Save, AlertTriangle, Loader2 } from 'lucide-react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { getDataset, updateDataset, UpdateDatasetRequest } from '@/api/datasets';
import { useToast } from '@/hooks/use-toast';
import { Switch } from '@/components/ui/switch';

const editDatasetSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100, 'Name is too long'),
  description: z.string().max(500, 'Description is too long').optional(),
  type: z.enum(['user_query', 'context', 'question_answer', 'conversation', 'custom']),
  is_public: z.boolean().default(false),
});

type EditDatasetFormValues = z.infer<typeof editDatasetSchema>;

export function DatasetEdit() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { toast } = useToast();

  const { data: dataset, isLoading, error } = useQuery({
    queryKey: ['dataset', id],
    queryFn: () => getDataset(id!),
    enabled: !!id,
  });

  const { register, handleSubmit, setValue, formState: { errors, isDirty } } = useForm<EditDatasetFormValues>({
    resolver: zodResolver(editDatasetSchema),
    defaultValues: {
      name: '',
      description: '',
      type: 'custom',
      is_public: false,
    },
  });

  // Set form values when dataset is loaded
  React.useEffect(() => {
    if (dataset) {
      setValue('name', dataset.name);
      setValue('description', dataset.description || '');
      setValue('type', dataset.type);
      setValue('is_public', dataset.is_public);
    }
  }, [dataset, setValue]);

  const updateDatasetMutation = useMutation({
    mutationFn: (data: UpdateDatasetRequest) => updateDataset(id!, data),
    onSuccess: (data) => {
      toast({
        title: 'Dataset Updated',
        description: `Successfully updated dataset: ${data.name}`,
      });
      queryClient.invalidateQueries({ queryKey: ['dataset', id] });
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      navigate(`/datasets/${id}`);
    },
    onError: (error) => {
      toast({
        title: 'Error Updating Dataset',
        description: 'Failed to update dataset. Please try again.',
        variant: 'destructive',
      });
      console.error('Dataset update error:', error);
    },
  });

  const onSubmit = (data: EditDatasetFormValues) => {
    updateDatasetMutation.mutate(data);
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <div className="h-8 w-48 bg-muted animate-pulse rounded"></div>
        </div>
        <Card>
          <CardHeader>
            <div className="h-7 w-64 bg-muted animate-pulse rounded mb-2"></div>
            <div className="h-5 w-full bg-muted animate-pulse rounded"></div>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {Array(4).fill(0).map((_, i) => (
                <div key={i} className="space-y-2">
                  <div className="h-5 w-32 bg-muted animate-pulse rounded"></div>
                  <div className="h-8 w-full bg-muted animate-pulse rounded"></div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error || !dataset) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <h1 className="text-3xl font-bold tracking-tight">Edit Dataset</h1>
        </div>
        <Card className="border-destructive">
          <CardHeader>
            <CardTitle className="flex items-center text-destructive">
              <AlertTriangle className="h-5 w-5 mr-2" />
              Error Loading Dataset
            </CardTitle>
          </CardHeader>
          <CardContent>
            <p>Failed to load dataset details. The dataset may not exist or there was a server error.</p>
          </CardContent>
          <CardFooter>
            <Button onClick={() => navigate('/datasets')}>
              Return to Datasets
            </Button>
          </CardFooter>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={() => navigate(-1)}>
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back
          </Button>
          <h1 className="text-3xl font-bold tracking-tight">Edit Dataset: {dataset.name}</h1>
        </div>
      </div>

      <Card>
        <form onSubmit={handleSubmit(onSubmit)}>
          <CardHeader>
            <CardTitle>Dataset Information</CardTitle>
            <CardDescription>
              Update the details for this dataset.
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-6">
            <div className="space-y-2">
              <Label htmlFor="name">Name</Label>
              <Input
                id="name"
                {...register('name')}
                placeholder="Enter a name for this dataset"
              />
              {errors.name && (
                <p className="text-sm text-destructive mt-1">{errors.name.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                {...register('description')}
                placeholder="Enter a description (optional)"
                rows={3}
              />
              {errors.description && (
                <p className="text-sm text-destructive mt-1">{errors.description.message}</p>
              )}
            </div>

            <div className="space-y-2">
              <Label htmlFor="type">Dataset Type</Label>
              <Select
                defaultValue={dataset.type}
                onValueChange={(value) => setValue('type', value as EditDatasetFormValues['type'])}
              >
                <SelectTrigger id="type">
                  <SelectValue placeholder="Select a dataset type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="user_query">User Query</SelectItem>
                  <SelectItem value="context">Context</SelectItem>
                  <SelectItem value="question_answer">Question & Answer</SelectItem>
                  <SelectItem value="conversation">Conversation</SelectItem>
                  <SelectItem value="custom">Custom</SelectItem>
                </SelectContent>
              </Select>
              {errors.type && (
                <p className="text-sm text-destructive mt-1">{errors.type.message}</p>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="is_public"
                checked={dataset.is_public}
                onCheckedChange={(checked) => setValue('is_public', checked)}
              />
              <Label htmlFor="is_public">Make this dataset public</Label>
            </div>

            <div className="p-4 bg-muted rounded-md">
              <h4 className="font-medium mb-2">Dataset File</h4>
              <p className="text-sm text-muted-foreground">
                Replacing the dataset file is not supported in this version. If you need to update the data,
                please create a new dataset.
              </p>
            </div>
          </CardContent>

          <CardFooter className="flex justify-between">
            <Button
              type="button"
              variant="outline"
              onClick={() => navigate(`/datasets/${id}`)}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={updateDatasetMutation.isPending || !isDirty}
            >
              {updateDatasetMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="mr-2 h-4 w-4" />
                  Save Changes
                </>
              )}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}