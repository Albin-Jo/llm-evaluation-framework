import React from 'react';
import { Outlet } from 'react-router-dom';
import { SidebarNav } from '@/components/sidebar-nav';
import { UserNav } from '@/components/user-nav';
import { ThemeToggle } from '@/components/theme-toggle';
import { Toaster } from '@/components/ui/toaster';

export function DashboardLayout() {
  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar - fixed width */}
      <div className="w-64 flex-shrink-0">
        <div className="flex flex-col h-full pt-5 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 overflow-y-auto">
          <div className="flex items-center flex-shrink-0 px-4 mb-5">
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">LLM Evaluation</h1>
          </div>
          <div className="flex-grow flex flex-col">
            <SidebarNav />
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col flex-grow overflow-hidden">
        {/* Top navigation */}
        <header className="border-b border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
          <div className="flex h-16 items-center px-4">
            <div className="md:hidden">
              {/* Mobile menu button would go here */}
            </div>

            <div className="ml-auto flex items-center gap-4">
              <ThemeToggle />
              <UserNav />
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-grow overflow-y-auto p-4 bg-gray-50 dark:bg-gray-950">
          <Outlet />
        </main>
      </div>

      {/* Toast notifications */}
      <Toaster />
    </div>
  );
}