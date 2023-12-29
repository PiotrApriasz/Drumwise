using System.Collections.Immutable;
using System.Security.Claims;
using Drumwise.API;
using Drumwise.API.ExceptionHandlers;
using Drumwise.Application.Common.Errors;
using Drumwise.Application.Common.Extensions;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Application.Common.Models.Identity;
using Drumwise.Infrastructure.Data;
using Drumwise.Infrastructure.Identity;
using FluentValidation;
using FluentValidation.Results;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.ConfigureIdentity(builder.Configuration);
builder.Services.AddApplicationServices(builder.Configuration);

builder.Services.AddExceptionHandler<ServerExceptionHandler>();
builder.Services.AddProblemDetails();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();

    await app.InitializeIdentityDatabaseAsync();
}

app.UseHttpsRedirection();

app.UseExceptionHandler();

app.MapIdentityApi<ApplicationUser>();

app.MapGet("/test",  async Task<IResult>
    ([FromServices] IIdentityService identityService) =>
{
    var result = await identityService
        .RegisterAdditionalUserData(new AdditionalUserDataRequest("test", "testowy", 0),
            new ClaimsPrincipal());

    return result.Match(
            onSuccess: result.ProduceSuccessApiResponse(),
            onFailure: result.ProduceErrorApiResponse());
})
.Produces(StatusCodes.Status204NoContent)
.ProducesValidationProblem(StatusCodes.Status400BadRequest)
.ProducesValidationProblem(StatusCodes.Status404NotFound);

app.Run();
