import { NgFor, NgIf, SlicePipe } from '@angular/common';
import { Component, ElementRef, EventEmitter, Input, OnInit, Output, SimpleChanges, ViewChild } from '@angular/core';
import { Router, RouterLink, RouterLinkActive, RouterModule } from '@angular/router';


interface Tab {
  title: string;
  icon: string;
  active: boolean;
  closable: boolean;
  dataurl: string;
}
@Component({
  selector: 'qrds-webheader',
  templateUrl: './qrecwebheader.component.html',
  styleUrls: ['./qrecwebheader.component.scss'],
  imports: [NgFor, NgIf, SlicePipe, RouterLink, RouterLinkActive, RouterModule]
})
export class QrecwebheaderComponent implements OnInit {
  @ViewChild('tabWrapPanel') tabWrapPanel!: ElementRef;
  @Input() envlabel = 'SIT';
  @Input() verlabel = 'SIT';
  showArrows = false;
  @Input() filtervalues: any[] = []; // The options passed from the parent component
  @Input() smartFilter = true; // Whether to enable smart filtering
  @Input() welcometxtlabel = 'Welcome';
  @Input() empname = 'Nadeem Ur-Rehman';
  @Input() empid = '146461';
  @Input() roletxtlabel = 'Role';
  @Input() userimg = '';
  @Input() roletxtvalue = 'Requestor';
  @Input() tabs: Tab[] = [];
  links: { txt: string; url: string }[] = [
    { txt: 'Link 1', url: 'https://example.com/1' },
    { txt: 'Link 2', url: 'https://example.com/2' }
  ];
  documents: { txt: string; url: string }[] = [
    { txt: 'Document 1', url: 'https://example.com/1' },
    { txt: 'Document 2', url: 'https://example.com/2' }
  ];
  @Input() issearchoptrequired = true;
  @Input() isquicklinkrequired = true;
  @Input() isdocumentrequired = true;
  @Input() isroletxtrequired = true;
  @Input() isProfilelinkrequired = true;
  @Input() dropdownOptions: string[] = [];

  @Output() formSubmit: EventEmitter<{ dropdownValue: string; textboxValue: string }> = new EventEmitter();
  @Output() menuClicked: EventEmitter<void> = new EventEmitter<void>();
  @Output() notificationClicked: EventEmitter<void> = new EventEmitter<void>();
  @Output() profilelinkClicked: EventEmitter<void> = new EventEmitter<void>();
  // @Output() logoutlinkClicked: EventEmitter<void> = new EventEmitter<void>();
  @Output() onClickLogOut = new EventEmitter();
  @Output() openDataEvent = new EventEmitter<boolean>();

  isDirty = false;

  filteredValues: any[] = [];
  searchText = '';
  selectedOptionIndex = -1; // For keyboard navigation
  showOptions = true; // Controls the visibility of the options list

  markAsDirty() {
    this.isDirty = true;
  }
  markAsClean() {
    this.isDirty = false;
  }


  ngAfterViewInit(): void {
    setTimeout(() => {
      this.updateArrowVisibility();
    }, 100); // A minimal delay

    window.addEventListener('resize', () => this.updateArrowVisibility());
  }

  scrollTabs(direction: string): void {
    const scrollAmount = 450; // Adjust based on your needs
    const currentScroll = this.tabWrapPanel.nativeElement.scrollLeft;
    this.tabWrapPanel.nativeElement.scrollTo({
      left: direction === 'left' ? currentScroll - scrollAmount : currentScroll + scrollAmount,
      behavior: 'smooth'
    });
    this.updateArrowVisibility();
  }

  updateArrowVisibility(): void {
    const container = this.tabWrapPanel.nativeElement;
    const tabsWidth = container.scrollWidth;
    const containerWidth = container.offsetWidth;
    this.showArrows = tabsWidth > containerWidth;
  }
  menuClick() {
    this.menuClicked.emit();
  }
  notificationClick() {
    this.notificationClicked.emit();
  }
  openUserData() {
    this.openDataEvent.emit(true);
  }
  profilelinkClick() {
    this.profilelinkClicked.emit();
  }
  logOutClicked() {
    // alert('logoutlinkClicked');
    // this.logoutlinkClicked.emit();
    this.onClickLogOut.emit();
  }

  selectedDropdownValue = '';
  textboxValue = '';

  removeTab(index: number) {
    const tab = this.tabs[index];
    if (tab.closable) {
      if (confirm('Are you sure you want to close this tab?')) {
        const wasActive = tab.active;
        this.tabs.splice(index, 1);
        console.log(this.tabs, wasActive, tab);
        if (wasActive && this.tabs.length > 0) {
          // If the closed tab was active, activate another tab.
          // Here we activate the last tab, but you can customize this logic.
          const lastTabIndex = this.tabs.length - 1;
          this.selectTab(lastTabIndex);
          const dataurlParts = this.tabs[lastTabIndex].dataurl.split('/');
          console.log({dataurlParts, lastTabIndex, wasActive, tab});
          this.router.navigate(['app', ...dataurlParts]);
        } else {
          this.router.navigate(['app']);
        }
      }
    }
    // Re-check arrow visibility
    setTimeout(() => {
      this.updateArrowVisibility();
    }, 0); // A minimal delay ensures DOM updates are reflected
  }
  selectTab(index: number) {
    this.tabs.forEach((tab, i) => tab.active = i === index);
  }

  constructor(private router: Router) { }

  ngOnInit(): void {
    this.filteredValues = this.filtervalues;
    window.addEventListener('resize', () => this.updateArrowVisibility());
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['dropdownOptions'] && this.dropdownOptions.length > 0) { this.selectedDropdownValue = this.dropdownOptions[0]; }
  }
  onSearch(): void {
    console.log('Dropdown Value:', this.selectedOptionIndex >= 0 ?
      this.filteredValues[this.selectedOptionIndex].name : 'None', 'Search Text:', this.searchText);
    // Add search logic here if necessary
    this.showOptions = false;
  }
  filterOptions(): void {
    if (!this.smartFilter) {
      this.filteredValues = this.filtervalues;
      return;
    }
    this.showOptions = true;
    this.filteredValues = this.filtervalues.filter(option =>
      option.name.toLowerCase().includes(this.searchText.toLowerCase())
    );
  }
  selectOption(option: any): void {
    this.searchText = option.name;
    this.showOptions = false; // Close the options list
    this.onSearch(); // Optionally trigger the search action
  }
  onKeydown(event: KeyboardEvent): void {
    if (event.key === 'ArrowDown' && this.selectedOptionIndex < this.filteredValues.length - 1) {
      this.selectedOptionIndex++;
    } else if (event.key === 'ArrowUp' && this.selectedOptionIndex > 0) {
      this.selectedOptionIndex--;
    } else if (event.key === 'Enter') {
      this.searchText = this.filteredValues[this.selectedOptionIndex].name;
      this.onSearch();
    }
  }
  closeOptions(): void {
    // Close the options list when the input loses focus
    setTimeout(() => this.showOptions = false, 200); // Delay allows for click selection to register
  }

  onSubmit() {
    this.formSubmit.emit({
      dropdownValue: this.selectedDropdownValue,
      textboxValue: this.textboxValue,
    });
  }

}
