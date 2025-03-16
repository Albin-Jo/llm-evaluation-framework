import React from 'react';
import { AppRoutes } from './routes';
import { Providers } from './providers/providers';

function App() {
  return (
    <Providers>
      <AppRoutes />
    </Providers>
  );
}

export default App;