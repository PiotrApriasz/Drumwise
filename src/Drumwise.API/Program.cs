using Drumwise.API;
using Drumwise.API.Endpoints;
using Drumwise.API.ExceptionHandlers;
using Drumwise.Infrastructure.Data;
using Drumwise.Infrastructure.Identity;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
// Learn more about configuring Swagger/OpenAPI at https://aka.ms/aspnetcore/swashbuckle
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

builder.Services.ConfigureIdentity(builder.Configuration);
builder.Services.AddApplicationServices(builder.Configuration);

builder.Services.AddExceptionHandler<BadHttpRequestExceptionHandler>();
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

// Map endpoints --------------------------------------------------
var apiEndpoints = app.MapGroup("/api");

apiEndpoints.MapIdentityApi<ApplicationUser>();
apiEndpoints.MapAdditionalIdentityEndpoints();
// ----------------------------------------------------------------

app.Run();
