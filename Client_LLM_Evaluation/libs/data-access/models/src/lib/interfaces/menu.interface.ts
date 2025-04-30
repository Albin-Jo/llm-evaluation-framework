export interface VerticalMenu extends Partial<CommonMenuItems> {
  submenu?: SubMenu[] | [];
}

export interface CommonMenuItems {
  name: string;
  dataurl?: string;
  id: string;
  icon?: string;
  roles?: [];
}

export interface SubMenu extends Partial<CommonMenuItems> {
  header: string;
  items: CommonMenuItems[]
}
