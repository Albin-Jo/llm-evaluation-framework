import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useMutation } from '@tanstack/react-query';
import { ChevronLeft, Upload, Info, Check, X, Loader2 } from 'lucide-react';
import { useForm } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { createDataset, CreateDatasetRequest } from '@/api/datasets';
import { useToast } from '@/hooks/use-toast';
import { Switch } from '@/components/ui/switch';

const createDatasetSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100, 'Name is too long'),
  description: z.string().max(500, 'Description is too long').optional(),
  type: z.enum(['user_query', 'context', 'question_answer', 'conversation', 'custom']),
  is_public: z.boolean().default(false),
  file: z.instanceof(File)
    .refine((file) => file.size <= 100 * 1024 * 1024, 'File size must be less than 100MB')
    .refine(
      (file) => ['text/csv', 'application/json', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
        .includes(file.type),
      'File must be CSV, JSON, or Excel'
    ),
});

type CreateDatasetFormValues = z.infer<typeof createDatasetSchema>;

export function DatasetCreate() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const { register, handleSubmit, setValue, formState: { errors }, watch } = useForm<CreateDatasetFormValues>({
    resolver: zodResolver(createDatasetSchema),
    defaultValues: {
      name: '',
      description: '',
      type: 'custom',
      is_public: false,
    },
  });

  const createDatasetMutation = useMutation({
    mutationFn: createDataset,
    onSuccess: (data) => {
      toast({
        title: 'Dataset Created',
        description: `Successfully created dataset: ${data.name}`,
      });
      navigate(`/datasets/${data.id}`);
    },
    onError: (error) => {
      toast({
        title: 'Error Creating Dataset',
        description: 'Failed to create dataset. Please try again.',
        variant: 'destructive',
      });
      console.error('Dataset creation error:', error);
    },
  });

  const onSubmit = (data: CreateDatasetFormValues) => {
    createDatasetMutation.mutate(data);
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] || null;
    setSelectedFile(file);
    setValue('file', file as File);
  };

  const fileSelected = watch('file');

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Button variant="outline" size="sm" asChild>
          <Link to="/datasets">
            <ChevronLeft className="h-4 w-4 mr-1" />
            Back to Datasets
          </Link>
        </Button>
        <h1 className="text-3xl font-bold tracking-tight">Create New Dataset</h1>
      </div>

      <Card>
        <form onSubmit={handleSubmit(onSubmit)}>
          <CardHeader>
            <CardTitle>Dataset Information</CardTitle>
            <CardDescription>
              Enter the details for your new dataset and upload the data file.
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
                defaultValue="custom"
                onValueChange={(value) => setValue('type', value as CreateDatasetFormValues['type'])}
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

            <div className="space-y-4">
              <Label htmlFor="file">Upload Dataset File</Label>

              <div className="flex items-center justify-center w-full">
                <label
                  htmlFor="file"
                  className="flex flex-col items-center justify-center w-full h-64 border-2 border-dashed rounded-lg cursor-pointer bg-muted/30 hover:bg-muted/50 transition-colors"
                >
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <Upload className="w-8 h-8 mb-4 text-muted-foreground" />
                    <p className="mb-2 text-sm text-muted-foreground">
                      <span className="font-semibold">Click to upload</span> or drag and drop
                    </p>
                    <p className="text-xs text-muted-foreground">
                      CSV, JSON, or Excel (max 100MB)
                    </p>
                    {selectedFile && (
                      <div className="mt-4 flex items-center gap-2 bg-primary-foreground px-3 py-2 rounded-md">
                        <Check className="h-4 w-4 text-green-500" />
                        <span className="text-sm font-medium">{selectedFile.name}</span>
                        <span className="text-xs text-muted-foreground">
                          ({(selectedFile.size / 1024 / 1024).toFixed(2)} MB)
                        </span>
                      </div>
                    )}
                  </div>
                  <input
                    id="file"
                    type="file"
                    accept=".csv,.json,.xls,.xlsx"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </label>
              </div>

              {errors.file && (
                <p className="text-sm text-destructive mt-1">{errors.file.message}</p>
              )}
            </div>

            <div className="flex items-center space-x-2">
              <Switch
                id="is_public"
                onCheckedChange={(checked) => setValue('is_public', checked)}
              />
              <Label htmlFor="is_public">Make this dataset public</Label>
            </div>
          </CardContent>

          <CardFooter className="flex justify-between">
            <Button
              type="button"
              variant="outline"
              onClick={() => navigate('/datasets')}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={createDatasetMutation.isPending || !fileSelected}
            >
              {createDatasetMutation.isPending ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Creating...
                </>
              ) : (
                'Create Dataset'
              )}
            </Button>
          </CardFooter>
        </form>
      </Card>
    </div>
  );
}