using System.Reflection;
using Ardalis.GuardClauses;
using Drumwise.Application.Common.Behaviours;
using Drumwise.Application.Common.Interfaces;
using Drumwise.Infrastructure.Data;
using Drumwise.Infrastructure.Data.Interceptors;
using Drumwise.Infrastructure.Identity;
using Drumwise.Infrastructure.Identity.Constants;
using FluentValidation;
using MediatR;
using MediatR.Pipeline;
using Microsoft.AspNetCore.Identity;
using Microsoft.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore.Diagnostics;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace Drumwise.DependencyInjection;

public static class ConfigureServices
{
    public static IServiceCollection AddInfrastructuresServices(this IServiceCollection services, IConfiguration configuration,
        string csName)
    {
        var connectionString = configuration.GetConnectionString(csName);
        Guard.Against.Null(connectionString, $"Connection string for '{csName}' not found");

        services.AddScoped<ISaveChangesInterceptor, AuditableEntityInterceptor>();
        services.AddScoped<ISaveChangesInterceptor, DispatchDomainEventsInterceptor>();

        services.AddDbContext<ApplicationDbContext>((sp, options) =>
        {
            options.AddInterceptors(sp.GetService<ISaveChangesInterceptor>()!);
            options.UseSqlite(connectionString);
        });

        services.AddScoped<IApplicationDbContext>(provider => provider.GetRequiredService<ApplicationDbContext>());
        services.AddScoped<ApplicationDbContextInitializer>();

        services.AddIdentity<ApplicationUser, Roles>()
            .AddRoles<IdentityRole>()
            .AddEntityFrameworkStores<ApplicationDbContext>();
        
        services.AddTransient<IIdentityService, IdentityService>();

        services.AddAuthorization(options =>
            options.AddPolicy(Policies.CanAddHomework, policy => policy.RequireRole(Roles.Teacher)));

        return services;
    }

    public static IServiceCollection AddApplicationServices(this IServiceCollection services)
    {
        services.AddAutoMapper(Assembly.GetExecutingAssembly());

        services.AddValidatorsFromAssembly(Assembly.GetExecutingAssembly());

        services.AddMediatR(cfg =>
        {
            cfg.RegisterServicesFromAssembly(Assembly.GetExecutingAssembly());
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(UnhandledExceptionBehaviour<,>));
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(AuthorizationBehaviour<,>));
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(ValidationBehaviour<,>));
            cfg.AddBehavior(typeof(IPipelineBehavior<,>), typeof(PerformanceBahaviour<,>));
            cfg.AddBehavior(typeof(IRequestPreProcessor<>), typeof(LoggingBehaviour<>));
        });

        return services;
    }
}