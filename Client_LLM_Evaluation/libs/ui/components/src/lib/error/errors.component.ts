import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { UpperCasePipe } from '@angular/common';

@Component({
    selector: 'app-error',
    templateUrl: './errors.component.html',
    styleUrls: ['./errors.component.scss'],
    imports: [RouterLink, UpperCasePipe]
})
export class ErrorsComponent implements OnInit {
  routeParams: any;
  data: any;

  constructor(
    private activatedRoute: ActivatedRoute,
  ) { }

  ngOnInit() {
    this.routeParams = this.activatedRoute.snapshot.queryParams;
    this.data = this.activatedRoute.snapshot.data;
  }
}
