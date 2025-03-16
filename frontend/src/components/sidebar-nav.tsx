import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  Database,
  LineChart,
  Brain,
  MessageSquare,
  GitCompare,
  Settings,
  BarChart2,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface NavItem {
  title: string;
  href: string;
  icon: React.ReactNode;
  isDeveloped?: boolean;
}

const navItems: NavItem[] = [
  {
    title: 'Dashboard',
    href: '/',
    icon: <BarChart2 className="h-5 w-5" />,
    isDeveloped: true,
  },
  {
    title: 'Datasets',
    href: '/datasets',
    icon: <Database className="h-5 w-5" />,
    isDeveloped: true,
  },
  {
    title: 'Evaluations',
    href: '/evaluations',
    icon: <LineChart className="h-5 w-5" />,
    isDeveloped: false,
  },
  {
    title: 'Micro Agents',
    href: '/microagents',
    icon: <Brain className="h-5 w-5" />,
    isDeveloped: false,
  },
  {
    title: 'Prompts',
    href: '/prompts',
    icon: <MessageSquare className="h-5 w-5" />,
    isDeveloped: false,
  },
  {
    title: 'Comparisons',
    href: '/comparisons',
    icon: <GitCompare className="h-5 w-5" />,
    isDeveloped: false,
  },
  {
    title: 'Settings',
    href: '/settings',
    icon: <Settings className="h-5 w-5" />,
    isDeveloped: false,
  },
];

export function SidebarNav() {
  const location = useLocation();

  return (
    <nav className="mt-2">
      {navItems.map((item) => {
        const isActive = location.pathname === item.href;

        return (
          <Link
            key={item.href}
            to={item.href}
            className={cn(
              "flex items-center px-4 py-2 mx-2 my-1 rounded-md text-sm font-medium",
              "transition-colors duration-200",
              isActive
                ? "bg-primary text-primary-foreground"
                : "text-gray-500 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-800"
            )}
          >
            <span className="mr-3 flex-shrink-0">{item.icon}</span>
            <span className="flex-grow">{item.title}</span>
            {!item.isDeveloped && (
              <span className="flex-shrink-0 text-xs bg-amber-100 dark:bg-amber-900 text-amber-800 dark:text-amber-200 px-2 py-0.5 rounded-full ml-2">
                Soon
              </span>
            )}
          </Link>
        );
      })}
    </nav>
  );
}