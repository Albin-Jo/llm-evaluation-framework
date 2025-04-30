/**
 * Define all endpoints here in this format
 */
export const API_ENDPOINTS = {
  user: {
    getAll: '/api/users',
    getById: (id: string) => `/api/users/${id}`,
    create: '/api/users/create',
  },

  /**
   * Usage of these endpoints:
   *
    -- getAll --
    getAllUsers() {
        return this.http.get(`${this.baseUrl}${API_ENDPOINTS.user.getAll}`);
    }

    -- getById --
    getUserById(id: string) {
      return this.http.get(`${this.baseUrl}${API_ENDPOINTS.user.getById(id)}`);
    }

   */

  userMeta: {
    getMenu: 'MenuService/api/MenuManager/GetUserRoleMenuItems',
  }
};
