import React from 'react';
import { useLocation } from 'react-router-dom';
import { DevelopmentNotice } from '@/components/development-notice';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

type PlaceholderConfig = {
  title: string;
  icon: React.ReactNode;
  description: string;
  section: string;
};

const placeholders: Record<string, PlaceholderConfig> = {
  '/evaluations': {
    title: 'Evaluations',
    icon: '📊',
    description: 'The evaluations module will allow you to run assessment experiments on various LLM models and track their performance metrics.',
    section: 'evaluations'
  },
  '/microagents': {
    title: 'Micro Agents',
    icon: '🤖',
    description: 'The micro agents module will allow you to configure and manage specialized agents for different evaluation tasks.',
    section: 'microagents'
  },
  '/prompts': {
    title: 'Prompts',
    icon: '💬',
    description: 'The prompts module will allow you to create, manage, and test prompt templates for your LLM evaluations.',
    section: 'prompts'
  },
  '/comparisons': {
    title: 'Comparisons',
    icon: '📈',
    description: 'The comparisons module will allow you to compare evaluation results side-by-side to identify strengths and weaknesses.',
    section: 'comparisons'
  },
  '/settings': {
    title: 'Settings',
    icon: '⚙️',
    description: 'The settings module will allow you to configure system preferences, manage users, and customize your evaluation environment.',
    section: 'settings'
  },
};

export function PlaceholderPage() {
  const location = useLocation();
  const path = location.pathname;

  // Find the exact match first
  let config = placeholders[path];

  // If no exact match, try to find a partial match
  if (!config) {
    const matchingPath = Object.keys(placeholders).find(p => path.startsWith(p));
    if (matchingPath) {
      config = placeholders[matchingPath];
    }
  }

  // Default fallback if no match found
  if (!config) {
    config = {
      title: 'Feature',
      icon: '🚧',
      description: 'This feature is currently under development.',
      section: 'feature'
    };
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight flex items-center gap-2">
        <span>{config.icon}</span> {config.title}
      </h1>

      <Card>
        <CardHeader>
          <CardTitle>{config.title} Module</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <DevelopmentNotice
            title={`${config.title} Module Under Development`}
            message={`This feature is currently under development and will be available soon.`}
          />

          <div className="mt-4">
            <h3 className="text-lg font-medium mb-2">About {config.title}</h3>
            <p>{config.description}</p>
          </div>

          <div className="mt-6">
            <h3 className="text-lg font-medium mb-2">Planned Features</h3>
            <ul className="list-disc list-inside space-y-1">
              {config.section === 'evaluations' && (
                <>
                  <li>Create and configure evaluation experiments</li>
                  <li>Choose from multiple evaluation methods (RAGAS, DeepEval, etc.)</li>
                  <li>View detailed evaluation results and metrics</li>
                  <li>Export evaluation reports</li>
                </>
              )}

              {config.section === 'microagents' && (
                <>
                  <li>Configure microservice-based agents</li>
                  <li>Monitor agent health and performance</li>
                  <li>Customize agent behavior</li>
                  <li>Test agents with sample inputs</li>
                </>
              )}

              {config.section === 'prompts' && (
                <>
                  <li>Create and manage prompt templates</li>
                  <li>Test prompts with variables</li>
                  <li>Organize prompts by category</li>
                  <li>Version control for prompts</li>
                </>
              )}

              {config.section === 'comparisons' && (
                <>
                  <li>Compare multiple evaluations side-by-side</li>
                  <li>Visualize performance differences</li>
                  <li>Generate comparative reports</li>
                  <li>Track improvements over time</li>
                </>
              )}

              {config.section === 'settings' && (
                <>
                  <li>Manage API connections</li>
                  <li>Configure system preferences</li>
                  <li>Manage user accounts and permissions</li>
                  <li>Set up notification preferences</li>
                </>
              )}

              {!['evaluations', 'microagents', 'prompts', 'comparisons', 'settings'].includes(config.section) && (
                <>
                  <li>Feature planning in progress</li>
                  <li>More details coming soon</li>
                </>
              )}
            </ul>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}