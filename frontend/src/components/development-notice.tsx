import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle } from 'lucide-react';

interface DevelopmentNoticeProps {
  title?: string;
  message?: string;
}

export function DevelopmentNotice({
  title = 'Feature Under Development',
  message = 'This feature is currently under development and will be available soon.',
}: DevelopmentNoticeProps) {
  return (
    <Card className="border-yellow-400 bg-yellow-100 dark:bg-yellow-800/50 shadow-md">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg font-semibold flex items-center gap-2 text-yellow-800 dark:text-yellow-300">
          <AlertTriangle className="h-5 w-5" />
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-4 text-yellow-700 dark:text-yellow-200">
        <p>{message}</p>
      </CardContent>
    </Card>
  );
}
